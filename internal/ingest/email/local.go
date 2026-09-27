package email

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/mail"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// maxFileBytes caps how much of one mail file is read; larger archives are truncated rather than risking
// memory exhaustion on an attacker-influenceable mailbox.
const maxFileBytes = 64 << 20

// LocalSource reads .eml messages and .mbox archives from a file or a directory tree, with no network or
// credentials.
type LocalSource struct {
	path string
}

// NewLocalSource builds a LocalSource over a file or directory path.
func NewLocalSource(path string) *LocalSource { return &LocalSource{path: path} }

// Name implements Source.
func (s *LocalSource) Name() string { return "local" }

// Messages implements Source: it parses the configured path and returns its
// messages, honouring opts.Since and opts.Max.
func (s *LocalSource) Messages(_ context.Context, opts Query) ([]Message, error) {
	info, err := os.Stat(s.path)
	if err != nil {
		return nil, fmt.Errorf("email: %w", err)
	}
	var files []string
	if info.IsDir() {
		err = filepath.WalkDir(s.path, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return nil //nolint:nilerr
			}
			if !d.IsDir() && isMailFile(p) {
				files = append(files, p)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		slices.Sort(files)
	} else {
		files = []string{s.path}
	}

	var msgs []Message
	for _, f := range files {
		data, err := readCapped(f, maxFileBytes)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", f, err)
		}
		if strings.EqualFold(filepath.Ext(f), ".mbox") || looksLikeMbox(data) {
			parsed, err := parseMbox(data)
			if err != nil {
				return nil, fmt.Errorf("parse mbox %s: %w", f, err)
			}
			msgs = append(msgs, parsed...)
			continue
		}
		m, err := parseMessage(data)
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", f, err)
		}
		msgs = append(msgs, m)
	}

	return applyQuery(msgs, opts), nil
}

// applyQuery filters parsed messages by Since and caps them at Max.
func applyQuery(msgs []Message, opts Query) []Message {
	if !opts.Since.IsZero() {
		msgs = slices.DeleteFunc(msgs, func(m Message) bool {
			return !m.Date.IsZero() && m.Date.Before(opts.Since)
		})
	}
	if opts.Max > 0 && len(msgs) > opts.Max {
		msgs = msgs[:opts.Max]
	}
	return msgs
}

// readCapped reads up to limit bytes of a file into memory.
func readCapped(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(io.LimitReader(f, limit))
}

// isMailFile reports whether a filename looks like an email file we can parse.
func isMailFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".eml", ".mbox", ".msg", ".txt":
		return true
	}
	return false
}

var mboxFromLine = []byte("From ")

// looksLikeMbox reports whether data begins with an mbox "From " envelope line.
func looksLikeMbox(data []byte) bool {
	return bytes.HasPrefix(data, mboxFromLine)
}

// utf8BOM is stripped before parsing: left in place it binds to the first header key and corrupts it.
var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

// parseMessage parses one RFC 5322 message into a Message.
func parseMessage(data []byte) (Message, error) {
	data = bytes.TrimPrefix(data, utf8BOM)
	msg, err := mail.ReadMessage(bytes.NewReader(data))
	if err != nil {
		return Message{}, err
	}
	h := msg.Header
	m := Message{
		MessageID: strings.TrimSpace(h.Get("Message-Id")),
		InReplyTo: strings.TrimSpace(h.Get("In-Reply-To")),
		Subject:   decodeHeaderWord(h.Get("Subject")),
		From:      firstAddress(h.Get("From")),
		To:        parseAddressList(h.Get("To")),
		Cc:        parseAddressList(h.Get("Cc")),
		Date:      parseDate(h),
		Body:      extractBody(h, msg.Body),
		Source:    "local",
	}
	return m, nil
}

// parseMbox splits an mbox archive on its "From " envelope lines and parses each
// contained message. Malformed individual messages are skipped, not fatal.
func parseMbox(data []byte) ([]Message, error) {
	var msgs []Message
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)

	var cur bytes.Buffer
	started := false
	flush := func() {
		if cur.Len() == 0 {
			return
		}
		if m, err := parseMessage(cur.Bytes()); err == nil {
			msgs = append(msgs, m)
		}
		cur.Reset()
	}
	for sc.Scan() {
		line := sc.Bytes()
		// An mbox envelope line ("From sender date") delimits messages; it is not
		// part of the RFC 5322 message itself, so we drop it.
		if bytes.HasPrefix(line, mboxFromLine) {
			flush()
			started = true
			continue
		}
		if !started {
			continue
		}
		// Unescape the ">From " mbox body quoting (best-effort, ">>From " etc.).
		if len(line) > 0 && line[0] == '>' && bytes.HasPrefix(bytes.TrimLeft(line, ">"), mboxFromLine) {
			line = line[1:]
		}
		cur.Write(line)
		cur.WriteByte('\n')
	}
	flush()
	return msgs, sc.Err()
}

// firstAddress parses the first mailbox from a header value (used for From).
func firstAddress(header string) Address {
	if as := parseAddressList(header); len(as) > 0 {
		return as[0]
	}
	return Address{}
}

// parseAddressList parses a comma-separated address header, decoding RFC 2047
// display names. On a parse failure it degrades to a single best-effort address.
func parseAddressList(header string) []Address {
	header = strings.TrimSpace(header)
	if header == "" {
		return nil
	}
	parsed, err := mail.ParseAddressList(header)
	if err != nil {
		// Fall back: treat the raw header as a bare address if it looks like one.
		if strings.Contains(header, "@") {
			return []Address{{Email: strings.Trim(header, "<> ")}}
		}
		return nil
	}
	out := make([]Address, 0, len(parsed))
	for _, a := range parsed {
		// ParseAddressList already RFC 2047-decodes names; decoding again would mangle a literal "=?...?=" name.
		out = append(out, Address{Name: a.Name, Email: a.Address})
	}
	return out
}

// parseDate reads the Date header, tolerating a few non-standard layouts before
// giving up (zero time).
func parseDate(h mail.Header) time.Time {
	if d, err := h.Date(); err == nil {
		return d.UTC()
	}
	raw := strings.TrimSpace(h.Get("Date"))
	if raw == "" {
		return time.Time{}
	}
	for _, layout := range []string{
		time.RFC1123Z, time.RFC1123, time.RFC822Z, time.RFC822,
		"2 Jan 2006 15:04:05 -0700", "Mon, 2 Jan 2006 15:04:05 -0700 (MST)",
	} {
		if d, err := time.Parse(layout, raw); err == nil {
			return d.UTC()
		}
	}
	return time.Time{}
}

var _ Source = (*LocalSource)(nil)
