package meeting

import (
	"testing"
	"time"
)

func TestDetectFormat(t *testing.T) {
	cases := []struct {
		name string
		data string
		want Format
	}{
		{"teams.vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi", FormatVTT},
		{"bom.vtt", string(utf8BOM) + "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi", FormatVTT},
		{"x.srt", "1\n00:00:00,000 --> 00:00:01,000\nAlice: hi", FormatSRT},
		{"notes.txt", "Alice: hello\nBob: hi", FormatPlain},
		{"noext-vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhi", FormatVTT},
		// Extension hint only, ambiguous content.
		{"hint.vtt", "just some text", FormatVTT},
	}
	for _, c := range cases {
		if got := DetectFormat(c.name, []byte(c.data)); got != c.want {
			t.Errorf("DetectFormat(%q)=%q want %q", c.name, got, c.want)
		}
	}
}

func TestParseVTTVoiceSpans(t *testing.T) {
	const vtt = `WEBVTT

00:00:00.000 --> 00:00:03.000
<v Alice>Let's talk about the retry policy</v>

00:00:03.000 --> 00:00:06.000
<v Alice>specifically in the fetch client.</v>

00:00:06.500 --> 00:00:09.000
<v Bob>Sounds good to me.</v>
`
	tr, err := Parse("standup.vtt", []byte(vtt))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Turns) != 2 {
		t.Fatalf("got %d turns, want 2 (coalesced); turns=%+v", len(tr.Turns), tr.Turns)
	}
	if tr.Turns[0].Speaker != "Alice" {
		t.Errorf("turn0 speaker=%q want Alice", tr.Turns[0].Speaker)
	}
	wantText := "Let's talk about the retry policy specifically in the fetch client."
	if tr.Turns[0].Text != wantText {
		t.Errorf("turn0 text=%q want %q", tr.Turns[0].Text, wantText)
	}
	if tr.Turns[0].End != 6*time.Second {
		t.Errorf("turn0 end=%v want 6s (coalesced)", tr.Turns[0].End)
	}
	if tr.Turns[1].Speaker != "Bob" || tr.Turns[1].Start != 6500*time.Millisecond {
		t.Errorf("turn1=%+v want Bob @6.5s", tr.Turns[1])
	}
	if tr.Title != "standup" {
		t.Errorf("title=%q want standup", tr.Title)
	}
}

func TestParseVTTColonSpeaker(t *testing.T) {
	const vtt = `WEBVTT

00:00:01.000 --> 00:00:04.000
Carol: We should add pagination here.
`
	tr, err := Parse("m.vtt", []byte(vtt))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Turns) != 1 || tr.Turns[0].Speaker != "Carol" {
		t.Fatalf("turns=%+v want one Carol turn", tr.Turns)
	}
	if tr.Turns[0].Text != "We should add pagination here." {
		t.Errorf("text=%q", tr.Turns[0].Text)
	}
}

func TestParseSRT(t *testing.T) {
	const srt = `1
00:00:00,000 --> 00:00:02,500
Dana: First point.

2
00:00:02,500 --> 00:00:05,000
Dana: Second point.

3
00:00:05,000 --> 00:00:07,000
Eve: Disagree.
`
	tr, err := Parse("rec.srt", []byte(srt))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Turns) != 2 {
		t.Fatalf("got %d turns want 2 coalesced: %+v", len(tr.Turns), tr.Turns)
	}
	if tr.Turns[0].Speaker != "Dana" || tr.Turns[0].Text != "First point. Second point." {
		t.Errorf("turn0=%+v", tr.Turns[0])
	}
	if tr.Turns[0].End != 5*time.Second {
		t.Errorf("turn0 end=%v want 5s", tr.Turns[0].End)
	}
}

func TestParsePlain(t *testing.T) {
	const txt = `[00:00:10] Frank: We decided to drop the cache layer.
It was causing stale reads.
Grace: Agreed, ship it.`
	tr, err := Parse("notes.txt", []byte(txt))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Turns) != 2 {
		t.Fatalf("got %d turns want 2: %+v", len(tr.Turns), tr.Turns)
	}
	if tr.Turns[0].Speaker != "Frank" || tr.Turns[0].Start != 10*time.Second {
		t.Errorf("turn0=%+v want Frank @10s", tr.Turns[0])
	}
	if tr.Turns[0].Text != "We decided to drop the cache layer. It was causing stale reads." {
		t.Errorf("turn0 text=%q (continuation not appended?)", tr.Turns[0].Text)
	}
	if tr.Turns[1].Speaker != "Grace" {
		t.Errorf("turn1 speaker=%q want Grace", tr.Turns[1].Speaker)
	}
}

func TestSplitSpeakerRejectsProse(t *testing.T) {
	// A sentence with a colon must not be misread as a speaker label.
	if sp, _ := splitSpeaker("Note: the following is important."); sp != "" {
		t.Errorf("splitSpeaker prose got speaker %q want none", sp)
	}
	if sp, _ := splitSpeaker("This is a long line, with commas: and more text"); sp != "" {
		t.Errorf("splitSpeaker comma-prose got speaker %q want none", sp)
	}
	if sp, txt := splitSpeaker("Alice: hi"); sp != "Alice" || txt != "hi" {
		t.Errorf("splitSpeaker(Alice: hi)=(%q,%q)", sp, txt)
	}
}

func TestParseTimecode(t *testing.T) {
	cases := []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		{"00:00:00.000", 0, true},
		{"01:02:03.500", time.Hour + 2*time.Minute + 3*time.Second + 500*time.Millisecond, true},
		{"02:05,250", 2*time.Minute + 5*time.Second + 250*time.Millisecond, true},
		{"00:01:30", 90 * time.Second, true},
		{"garbage", 0, false},
		{"", 0, false},
	}
	for _, c := range cases {
		got, ok := parseTimecode(c.in)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("parseTimecode(%q)=(%v,%v) want (%v,%v)", c.in, got, ok, c.want, c.ok)
		}
	}
}

func TestParseEmptyErrors(t *testing.T) {
	if _, err := Parse("empty.txt", []byte("   \n\n")); err == nil {
		t.Error("expected error for empty transcript")
	}
}
