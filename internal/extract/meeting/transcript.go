// Package meeting ingests meeting transcripts — the formats Microsoft Teams,
// Zoom, and whisper.cpp all export — into chiton's knowledge graph and links
// the decisions made in them back to the code symbols they reference.
//
// Transcript parsing is pure-Go and dependency-free: a meeting downloaded as a
// .vtt from Teams, a .srt produced by whisper.cpp, or a hand-written
// "Speaker: text" notes file all flow through the same Parse path into a
// normalized Transcript. The upstream sources (internal/ingest/teams for the
// Microsoft Graph transcript API, internal/ingest/audio for whisper
// transcription) produce these files; this package is where they converge into
// graph nodes (ToGraph) and where the meeting→code linkage subagent flow lives
// (LinkService).
package meeting

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
	"time"
)

// Turn is one speaker turn in a transcript. Start/End are offsets from the
// meeting start; they are zero when the source format carries no timing (a
// plain notes file). Speaker is "" when the line had no attribution.
type Turn struct {
	Speaker string
	Start   time.Duration
	End     time.Duration
	Text    string
}

// Transcript is a parsed meeting, source-format-agnostic.
type Transcript struct {
	// Title is a human label for the meeting (defaults to the file base name).
	Title string
	// SourceURI identifies the meeting; ToGraph defaults it to meeting://<slug>
	// when empty. Teams sets teams://..., file ingest sets file://...
	SourceURI string
	// Date is the meeting date when known (zero otherwise).
	Date  time.Time
	Turns []Turn
}

// Format identifies a transcript serialization.
type Format string

const (
	FormatVTT   Format = "vtt"
	FormatSRT   Format = "srt"
	FormatPlain Format = "plain"
)

// utf8BOM is the UTF-8 byte-order-mark prefix that Windows-authored and some
// Teams-exported transcripts carry. Declared as explicit bytes (not a literal)
// so the source file itself stays BOM-free.
var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

// proseLeadIn is the set of common line-leading words that share the
// "Word: text" shape with a speaker label but are prose, not names. They are
// rejected by splitSpeaker so a notes file does not sprout phantom speakers.
var proseLeadIn = map[string]bool{
	"note": true, "notes": true, "todo": true, "fixme": true, "warning": true,
	"action": true, "actions": true, "decision": true, "decisions": true,
	"summary": true, "agenda": true, "attendees": true, "present": true,
	"apologies": true, "ps": true, "nb": true, "http": true, "https": true,
	"topic": true, "subject": true, "date": true, "time": true, "location": true,
}

// DetectFormat sniffs the transcript format from its content (and, as a weak
// tiebreaker, a filename hint). The WEBVTT magic header is authoritative; an
// SRT is recognised by its "N\n HH:MM:SS,mmm --> HH:MM:SS,mmm" cue shape;
// everything else is treated as a plain "Speaker: text" transcript.
func DetectFormat(name string, data []byte) Format {
	trimmed := bytes.TrimLeft(bytes.TrimPrefix(data, utf8BOM), " \t\r\n")
	if bytes.HasPrefix(trimmed, []byte("WEBVTT")) {
		return FormatVTT
	}
	if looksLikeSRT(trimmed) {
		return FormatSRT
	}
	lower := strings.ToLower(name)
	switch {
	case strings.HasSuffix(lower, ".vtt"):
		return FormatVTT
	case strings.HasSuffix(lower, ".srt"):
		return FormatSRT
	}
	return FormatPlain
}

// Parse parses a transcript of any supported format. name is a filename hint
// used for format detection and as the default title.
func Parse(name string, data []byte) (*Transcript, error) {
	var turns []Turn
	switch DetectFormat(name, data) {
	case FormatVTT:
		turns = parseVTT(data)
	case FormatSRT:
		turns = parseSRT(data)
	default:
		turns = parsePlain(data)
	}
	turns = coalesce(turns)
	if len(turns) == 0 {
		return nil, fmt.Errorf("meeting: no speaker turns found in %q", name)
	}
	return &Transcript{Title: titleFromName(name), Turns: turns}, nil
}

// looksLikeSRT reports whether data opens with an SRT cue: an index line
// followed by a "timecode --> timecode" line using comma millisecond separators.
func looksLikeSRT(data []byte) bool {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	// Find the first non-empty line; it should be a bare integer index.
	var first string
	for sc.Scan() {
		first = strings.TrimSpace(sc.Text())
		if first != "" {
			break
		}
	}
	if first == "" || !isAllDigits(first) {
		return false
	}
	if !sc.Scan() {
		return false
	}
	line := strings.TrimSpace(sc.Text())
	return strings.Contains(line, "-->") && strings.Contains(line, ",")
}

// --- WebVTT ---

// parseVTT parses WebVTT cues. Speaker attribution comes from a <v Name>...
// voice span when present (Teams), else a leading "Name: " on the payload.
func parseVTT(data []byte) []Turn {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	var turns []Turn
	var start, end time.Duration
	var inCue bool
	var payload []string

	flush := func() {
		if inCue {
			speaker, text := splitSpeaker(strings.Join(payload, " "))
			if text != "" {
				turns = append(turns, Turn{Speaker: speaker, Start: start, End: end, Text: text})
			}
		}
		inCue = false
		payload = payload[:0]
	}

	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		trimmed := strings.TrimSpace(line)
		if strings.Contains(line, "-->") {
			flush()
			s, e, ok := parseCueTiming(line)
			if !ok {
				continue
			}
			start, end, inCue = s, e, true
			continue
		}
		if trimmed == "" {
			flush()
			continue
		}
		if !inCue {
			// Header lines (WEBVTT, NOTE, STYLE), cue identifiers, region defs:
			// ignore anything outside a cue body.
			continue
		}
		payload = append(payload, stripVTTTags(trimmed))
	}
	flush()
	return turns
}

// --- SRT ---

// parseSRT parses SubRip cues. Speaker attribution, when present, is a leading
// "Name: " on the cue text (whisper.cpp emits unattributed SRT; some tools add
// speaker prefixes).
func parseSRT(data []byte) []Turn {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	var turns []Turn
	var start, end time.Duration
	var inCue bool
	var payload []string

	flush := func() {
		if inCue {
			speaker, text := splitSpeaker(strings.Join(payload, " "))
			if text != "" {
				turns = append(turns, Turn{Speaker: speaker, Start: start, End: end, Text: text})
			}
		}
		inCue = false
		payload = payload[:0]
	}

	for sc.Scan() {
		line := strings.TrimSpace(strings.TrimRight(sc.Text(), "\r"))
		if strings.Contains(line, "-->") {
			s, e, ok := parseCueTiming(line)
			if !ok {
				continue
			}
			flush()
			start, end, inCue = s, e, true
			continue
		}
		if line == "" {
			flush()
			continue
		}
		if !inCue {
			// A bare index line between cues — skip it.
			if isAllDigits(line) {
				continue
			}
			continue
		}
		payload = append(payload, stripVTTTags(line))
	}
	flush()
	return turns
}

// --- plain "Speaker: text" ---

// parsePlain parses a free-form transcript where each turn is "Speaker: text",
// optionally prefixed by a [timestamp]. Lines without a "Speaker:" prefix are
// appended to the current turn as continuations.
func parsePlain(data []byte) []Turn {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	var turns []Turn
	for sc.Scan() {
		raw := strings.TrimSpace(strings.TrimRight(sc.Text(), "\r"))
		if raw == "" {
			continue
		}
		ts, rest := stripLeadingTimestamp(raw)
		speaker, text := splitSpeaker(rest)
		if speaker == "" && len(turns) > 0 {
			// Continuation of the previous turn.
			prev := &turns[len(turns)-1]
			prev.Text = strings.TrimSpace(prev.Text + " " + text)
			continue
		}
		if text == "" {
			continue
		}
		turns = append(turns, Turn{Speaker: speaker, Start: ts, Text: text})
	}
	return turns
}

// --- shared helpers ---

// coalesce merges consecutive turns that share a speaker (common in VTT/SRT
// where one utterance spans several timed cues), extending the End and joining
// the text. Turns with no speaker are never merged into a speaker's turn.
func coalesce(turns []Turn) []Turn {
	if len(turns) == 0 {
		return turns
	}
	out := turns[:0:0]
	for _, t := range turns {
		if n := len(out); n > 0 && t.Speaker != "" && out[n-1].Speaker == t.Speaker {
			out[n-1].Text = strings.TrimSpace(out[n-1].Text + " " + t.Text)
			if t.End > out[n-1].End {
				out[n-1].End = t.End
			}
			continue
		}
		out = append(out, t)
	}
	return out
}

// splitSpeaker extracts a leading "Name: " speaker label from a line. It only
// treats the prefix as a speaker when the name is short and plausibly a name
// (no sentence-ending punctuation), so a normal sentence containing a colon
// ("Note: the following") is not misread as a speaker named "Note".
func splitSpeaker(s string) (speaker, text string) {
	s = strings.TrimSpace(s)
	idx := strings.Index(s, ":")
	if idx <= 0 || idx > 48 {
		return "", s
	}
	name := strings.TrimSpace(s[:idx])
	rest := strings.TrimSpace(s[idx+1:])
	if name == "" || rest == "" {
		return "", s
	}
	// Reject prefixes that look like prose rather than a speaker label.
	if strings.ContainsAny(name, ".!?,;\"") {
		return "", s
	}
	// A speaker label is at most a few words (most transcript speaker labels are
	// plain "First Last"); longer prefixes are sentences, not names.
	if wordCount(name) > 5 {
		return "", s
	}
	// Reject the common prose lead-in words that share the "Word: text" shape
	// (a notes file line "Note: ship it" must not invent a speaker named Note).
	if proseLeadIn[strings.ToLower(name)] {
		return "", s
	}
	return name, rest
}

// stripVTTTags removes inline VTT/SRT markup: <v Name> voice spans (the name is
// hoisted to the front as "Name: " so splitSpeaker can pick it up), <c> class
// spans, <i>/<b> styling, and <00:00:00.000> inline timestamps.
func stripVTTTags(s string) string {
	if !strings.Contains(s, "<") {
		return s
	}
	var b strings.Builder
	var voice string
	i := 0
	for i < len(s) {
		c := s[i]
		if c != '<' {
			b.WriteByte(c)
			i++
			continue
		}
		j := strings.IndexByte(s[i:], '>')
		if j < 0 {
			b.WriteByte(c)
			i++
			continue
		}
		tag := s[i+1 : i+j]
		if v, ok := voiceName(tag); ok {
			voice = v
		}
		i += j + 1
	}
	out := strings.TrimSpace(b.String())
	if voice != "" && !strings.HasPrefix(out, voice+":") {
		return voice + ": " + out
	}
	return out
}

// voiceName returns the speaker from a "v Name" voice-span tag, e.g. <v Alice>
// or <v.loud Jane Doe>. The name is everything after the first space; any
// classes (".loud") attach to the leading "v" token before that space.
func voiceName(tag string) (string, bool) {
	tag = strings.TrimSpace(tag)
	if tag != "v" && !strings.HasPrefix(tag, "v ") && !strings.HasPrefix(tag, "v.") {
		return "", false
	}
	_, name, ok := strings.Cut(tag, " ")
	name = strings.TrimSpace(name)
	if !ok || name == "" {
		return "", false // "<v>" or "<v.loud>" with no name
	}
	return name, true
}

// parseCueTiming parses a "HH:MM:SS.mmm --> HH:MM:SS.mmm" (or ,mmm) cue header,
// tolerating trailing cue settings after the end timecode.
func parseCueTiming(line string) (start, end time.Duration, ok bool) {
	parts := strings.SplitN(line, "-->", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	s, ok1 := parseTimecode(strings.TrimSpace(parts[0]))
	endField := strings.TrimSpace(parts[1])
	// Drop any cue settings ("align:start position:50%") after the timecode.
	if sp := strings.IndexAny(endField, " \t"); sp >= 0 {
		endField = endField[:sp]
	}
	e, ok2 := parseTimecode(endField)
	if !ok1 || !ok2 {
		return 0, 0, false
	}
	return s, e, true
}

// parseTimecode parses HH:MM:SS.mmm or MM:SS.mmm (with '.' or ',' for the
// fractional separator) into a Duration.
func parseTimecode(s string) (time.Duration, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	s = strings.Replace(s, ",", ".", 1)
	secPart := s
	var millis time.Duration
	if dot := strings.LastIndex(s, "."); dot >= 0 {
		frac := s[dot+1:]
		secPart = s[:dot]
		// Normalise to milliseconds (pad/truncate to 3 digits).
		for len(frac) < 3 {
			frac += "0"
		}
		frac = frac[:3]
		ms, err := atoiStrict(frac)
		if err != nil {
			return 0, false
		}
		millis = time.Duration(ms) * time.Millisecond
	}
	fields := strings.Split(secPart, ":")
	var h, m, sec int
	switch len(fields) {
	case 3:
		var err error
		if h, err = atoiStrict(fields[0]); err != nil {
			return 0, false
		}
		if m, err = atoiStrict(fields[1]); err != nil {
			return 0, false
		}
		if sec, err = atoiStrict(fields[2]); err != nil {
			return 0, false
		}
	case 2:
		var err error
		if m, err = atoiStrict(fields[0]); err != nil {
			return 0, false
		}
		if sec, err = atoiStrict(fields[1]); err != nil {
			return 0, false
		}
	default:
		return 0, false
	}
	d := time.Duration(h)*time.Hour + time.Duration(m)*time.Minute +
		time.Duration(sec)*time.Second + millis
	return d, true
}

// stripLeadingTimestamp removes a leading "[HH:MM:SS]" / "(MM:SS)" / "HH:MM:SS"
// timestamp from a plain-transcript line, returning the parsed offset and the
// remainder. Returns (0, line) when there is no recognisable timestamp.
func stripLeadingTimestamp(s string) (time.Duration, string) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, s
	}
	open := s[0]
	if open == '[' || open == '(' {
		closeCh := byte(']')
		if open == '(' {
			closeCh = ')'
		}
		if end := strings.IndexByte(s, closeCh); end > 0 {
			if d, ok := parseTimecode(s[1:end]); ok {
				return d, strings.TrimSpace(s[end+1:])
			}
		}
		return 0, s
	}
	// Bare leading timecode followed by whitespace: "00:01:23 Alice: ...".
	if sp := strings.IndexAny(s, " \t"); sp > 0 {
		if d, ok := parseTimecode(s[:sp]); ok {
			return d, strings.TrimSpace(s[sp:])
		}
	}
	return 0, s
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// atoiStrict parses a non-negative integer, rejecting empty/garbage input.
func atoiStrict(s string) (int, error) {
	if !isAllDigits(s) {
		return 0, fmt.Errorf("not an integer: %q", s)
	}
	n := 0
	for i := 0; i < len(s); i++ {
		n = n*10 + int(s[i]-'0')
	}
	return n, nil
}

func wordCount(s string) int {
	return len(strings.Fields(s))
}

// titleFromName derives a default meeting title from a filename or URI, dropping
// directories and the extension.
func titleFromName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "meeting"
	}
	if i := strings.LastIndexAny(name, "/\\"); i >= 0 {
		name = name[i+1:]
	}
	if dot := strings.LastIndex(name, "."); dot > 0 {
		name = name[:dot]
	}
	if name == "" {
		return "meeting"
	}
	return name
}
