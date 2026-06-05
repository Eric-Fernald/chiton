//go:build whisper_cgo

// This file implements chiton's in-process whisper.cpp transcription backend.
// It is EXCLUDED from the default build: it needs CGO_ENABLED=1, a C compiler,
// a built libwhisper, and the binding module, none of which the pure-Go default
// build requires. To use it:
//
//	go get github.com/ggerganov/whisper.cpp/bindings/go
//	CGO_ENABLED=1 go build -tags whisper_cgo ./...
//
// then call audio.NewCgo(cfg). It accepts a 16 kHz mono PCM WAV file (convert
// other formats first, e.g. `ffmpeg -i in.mp4 -ar 16000 -ac 1 out.wav`). The
// exact whisper binding API drifts between whisper.cpp releases; adjust the
// Process call if your pinned version differs.
package audio

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	wsp "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
)

// maxWAVChunkBytes caps a single WAV chunk to bound memory against a malformed
// or hostile header (~1 GiB ≈ 90 min of 16 kHz mono 16-bit PCM).
const maxWAVChunkBytes = 1 << 30

// CgoTranscriber runs whisper.cpp in-process via its Go bindings.
type CgoTranscriber struct {
	cfg Config
}

// NewCgo builds an in-process whisper.cpp transcriber. cfg.Model must be a ggml
// model path.
func NewCgo(cfg Config) (*CgoTranscriber, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("whisper_cgo: cfg.Model (ggml model path) is required")
	}
	return &CgoTranscriber{cfg: cfg}, nil
}

// Transcribe loads the model, decodes a 16 kHz mono WAV, and runs whisper,
// collecting segments into a meeting.Transcript (one turn per segment).
func (c *CgoTranscriber) Transcribe(_ context.Context, audioPath string) (*meeting.Transcript, error) {
	samples, err := readWAVMono16k(audioPath)
	if err != nil {
		return nil, err
	}
	model, err := wsp.New(c.cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", c.cfg.Model, err)
	}
	defer model.Close()

	wctx, err := model.NewContext()
	if err != nil {
		return nil, err
	}
	if c.cfg.Language != "" {
		if err := wctx.SetLanguage(c.cfg.Language); err != nil {
			return nil, err
		}
	}
	if err := wctx.Process(samples, nil, nil, nil); err != nil {
		return nil, fmt.Errorf("whisper process: %w", err)
	}

	tr := &meeting.Transcript{
		Title: strings.TrimSuffix(filepath.Base(audioPath), filepath.Ext(audioPath)),
	}
	for {
		seg, err := wctx.NextSegment()
		if err != nil {
			break // io.EOF when segments are exhausted
		}
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		tr.Turns = append(tr.Turns, meeting.Turn{
			Start: seg.Start,
			End:   seg.End,
			Text:  text,
		})
	}
	if len(tr.Turns) == 0 {
		return nil, fmt.Errorf("whisper produced no segments for %s", audioPath)
	}
	return tr, nil
}

// readWAVMono16k decodes a 16-bit PCM, mono, 16 kHz WAV file to []float32 in
// [-1,1]. It is intentionally minimal: convert other formats/rates with ffmpeg
// before calling. A non-conforming file is rejected with a clear error.
func readWAVMono16k(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var riff [12]byte
	if _, err := io.ReadFull(f, riff[:]); err != nil {
		return nil, err
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s is not a WAV file", path)
	}

	var dataChunk []byte
	var channels, bits int
	var sampleRate uint32
	for {
		var hdr [8]byte
		if _, err := io.ReadFull(f, hdr[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return nil, err
		}
		id := string(hdr[0:4])
		size := binary.LittleEndian.Uint32(hdr[4:8])
		// Guard against a malformed/hostile header declaring a huge chunk: a
		// 0xFFFFFFFF size would otherwise make([]byte, ~4GiB) and OOM before any
		// format validation. 1 GiB bounds ~90 min of 16 kHz mono 16-bit audio.
		if size > maxWAVChunkBytes {
			return nil, fmt.Errorf("%s: WAV chunk %q declares %d bytes (> %d cap)", path, id, size, maxWAVChunkBytes)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("%s: WAV fmt chunk too small (%d bytes)", path, size)
			}
			buf := make([]byte, size)
			if _, err := io.ReadFull(f, buf); err != nil {
				return nil, err
			}
			channels = int(binary.LittleEndian.Uint16(buf[2:4]))
			sampleRate = binary.LittleEndian.Uint32(buf[4:8])
			bits = int(binary.LittleEndian.Uint16(buf[14:16]))
		case "data":
			buf := make([]byte, size)
			if _, err := io.ReadFull(f, buf); err != nil {
				return nil, err
			}
			dataChunk = buf
		default:
			if _, err := f.Seek(int64(size), io.SeekCurrent); err != nil {
				return nil, err
			}
		}
		if size%2 == 1 { // chunks are word-aligned
			if _, err := f.Seek(1, io.SeekCurrent); err != nil && err != io.EOF {
				return nil, err
			}
		}
	}

	if channels != 1 || bits != 16 || sampleRate != 16000 {
		return nil, fmt.Errorf("%s must be 16 kHz mono 16-bit PCM (got %d Hz, %d ch, %d-bit); "+
			"convert with: ffmpeg -i <in> -ar 16000 -ac 1 -c:a pcm_s16le out.wav", path, sampleRate, channels, bits)
	}
	if dataChunk == nil {
		return nil, fmt.Errorf("%s has no data chunk", path)
	}

	n := len(dataChunk) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		s := int16(binary.LittleEndian.Uint16(dataChunk[2*i : 2*i+2]))
		out[i] = float32(s) / 32768.0
	}
	return out, nil
}
