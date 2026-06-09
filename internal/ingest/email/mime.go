package email

import (
	"bytes"
	"encoding/base64"
	"io"
	"mime"
	"mime/multipart"
	"mime/quotedprintable"
	"net/mail"
	"strings"

	"golang.org/x/net/html"
)

// maxMIMEDepth bounds recursion into nested multipart bodies (defensive against
// pathological/hostile messages).
const maxMIMEDepth = 10

// extractBody renders an RFC 5322 message body to plain text: it decodes the
// transfer encoding, walks multipart containers preferring text/plain, and
// strips HTML when only an HTML alternative is present. charset handling is
// best-effort UTF-8 (the overwhelmingly common case for modern mail).
func extractBody(header mail.Header, body io.Reader) string {
	raw, _ := io.ReadAll(body)
	plain, htmlText := collectParts(header.Get("Content-Type"), header.Get("Content-Transfer-Encoding"), raw, 0)
	if s := strings.TrimSpace(strings.Join(plain, "\n\n")); s != "" {
		return s
	}
	return strings.TrimSpace(strings.Join(htmlText, "\n\n"))
}

// collectParts decodes one MIME entity, recursing into multipart containers, and
// returns the text/plain bodies and the (HTML-stripped) text/html bodies it
// found, each in document order.
func collectParts(contentType, cte string, raw []byte, depth int) (plain, htmlText []string) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType == "" {
		mediaType = "text/plain" // RFC 2045 default
	}

	if strings.HasPrefix(mediaType, "multipart/") && depth < maxMIMEDepth {
		boundary := params["boundary"]
		if boundary == "" {
			return nil, nil
		}
		mr := multipart.NewReader(bytes.NewReader(raw), boundary)
		for {
			part, err := mr.NextPart()
			if err != nil {
				break
			}
			partRaw, _ := io.ReadAll(part)
			p, h := collectParts(part.Header.Get("Content-Type"), part.Header.Get("Content-Transfer-Encoding"), partRaw, depth+1)
			plain = append(plain, p...)
			htmlText = append(htmlText, h...)
			_ = part.Close()
		}
		return plain, htmlText
	}

	decoded := decodeTransferEncoding(raw, cte)
	switch {
	case mediaType == "text/html":
		if t := strings.TrimSpace(htmlToText(string(decoded))); t != "" {
			htmlText = append(htmlText, t)
		}
	case strings.HasPrefix(mediaType, "text/"):
		if t := strings.TrimSpace(string(decoded)); t != "" {
			plain = append(plain, t)
		}
	}
	return plain, htmlText
}

// decodeTransferEncoding decodes a Content-Transfer-Encoding'd body. Unknown or
// identity encodings (7bit/8bit/binary) pass through unchanged; a decode error
// falls back to the raw bytes rather than dropping content.
func decodeTransferEncoding(raw []byte, cte string) []byte {
	switch strings.ToLower(strings.TrimSpace(cte)) {
	case "base64":
		clean := stripWhitespace(raw)
		if dec, err := base64.StdEncoding.DecodeString(string(clean)); err == nil {
			return dec
		}
		if dec, err := base64.RawStdEncoding.DecodeString(string(clean)); err == nil {
			return dec
		}
		return raw
	case "quoted-printable":
		if dec, err := io.ReadAll(quotedprintable.NewReader(bytes.NewReader(raw))); err == nil {
			return dec
		}
		return raw
	default:
		return raw
	}
}

func stripWhitespace(b []byte) []byte {
	out := make([]byte, 0, len(b))
	for _, c := range b {
		switch c {
		case ' ', '\t', '\r', '\n':
		default:
			out = append(out, c)
		}
	}
	return out
}

// htmlToText extracts visible text from an HTML document, dropping <script> and
// <style> content and collapsing runs of whitespace. Mirrors the URL ingester's
// guarantee that script content never leaks into extracted text.
func htmlToText(s string) string {
	z := html.NewTokenizer(strings.NewReader(s))
	var b strings.Builder
	skip := 0
	for {
		switch z.Next() {
		case html.ErrorToken:
			return strings.Join(strings.Fields(b.String()), " ")
		case html.StartTagToken:
			if n, _ := z.TagName(); isSkippedTag(n) {
				skip++
			}
		case html.EndTagToken:
			if n, _ := z.TagName(); isSkippedTag(n) && skip > 0 {
				skip--
			}
		case html.TextToken:
			if skip == 0 {
				b.Write(z.Text())
				b.WriteByte(' ')
			}
		}
	}
}

func isSkippedTag(name []byte) bool {
	switch string(name) {
	case "script", "style":
		return true
	}
	return false
}

// decodeHeaderWord decodes RFC 2047 encoded-words (=?utf-8?B?...?=) in a header
// value such as a Subject or a display name, returning the input unchanged when
// it contains no encoded words or fails to decode.
func decodeHeaderWord(s string) string {
	dec := new(mime.WordDecoder)
	if out, err := dec.DecodeHeader(s); err == nil {
		return out
	}
	return s
}
