// Package audio transcribes local audio/video into a meeting.Transcript.
//
// The default, always-available backend (NewShell) shells out to a configured
// whisper CLI (OpenAI's `whisper` or whisper.cpp's `whisper-cli`) — pure Go,
// no CGo, no build-time C toolchain. An in-process whisper.cpp backend lives in
// whisper_cgo.go behind the `whisper_cgo` build tag (NewCgo); it needs
// CGO_ENABLED=1, a C compiler, libwhisper, and
// `go get github.com/ggerganov/whisper.cpp/bindings/go`, so it is excluded from
// the default build. Both backends produce the same Transcript, which then
// flows through meeting.ToGraph exactly like a downloaded .vtt.
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
