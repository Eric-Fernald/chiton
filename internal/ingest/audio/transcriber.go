// Package audio transcribes local audio or video into a meeting.Transcript. It shells out to a whisper CLI
// (NewShell, pure Go) or, with the whisper_cgo build tag, uses in-process whisper.cpp bindings (NewCgo).
package audio

import (
	"context"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
)

// Config tunes a Transcriber.
type Config struct {
	// Binary is the whisper CLI to invoke; empty auto-detects a known one.
	Binary string
	// Model is the model name (OpenAI whisper, e.g. "base") or the ggml model
	// path (whisper.cpp). Required for whisper.cpp.
	Model string
	// Language is an ISO code (e.g. "en"); empty lets whisper auto-detect.
	Language string
}

// Transcriber turns an audio/video file into a parsed transcript.
type Transcriber interface {
	// Transcribe transcribes audioPath. The returned Transcript has its turns
	// populated; the caller sets Title/SourceURI/Date as appropriate.
	Transcribe(ctx context.Context, audioPath string) (*meeting.Transcript, error)
}

// New returns the default pure-Go shell-out transcriber. Build with
// -tags whisper_cgo to also get NewCgo (in-process whisper.cpp bindings).
func New(cfg Config) Transcriber {
	return NewShell(cfg)
}
