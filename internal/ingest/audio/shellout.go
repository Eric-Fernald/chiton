package audio

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
)

// knownBinaries are the whisper CLIs ShellTranscriber auto-detects, in
// preference order.
var knownBinaries = []string{"whisper", "whisper-cli", "whisper-cpp", "whisper.cpp"}

// ShellTranscriber transcribes by invoking a whisper CLI and parsing its WebVTT output.
// Tests can swap in their own run func, so they need no real whisper install.
type ShellTranscriber struct {
	cfg Config
	// run executes the command; defaults to exec.CommandContext. Tests override
	// it to fake whisper by writing a .vtt into the output directory.
	run func(ctx context.Context, name, dir string, args []string) error
}

// NewShell builds a ShellTranscriber.
func NewShell(cfg Config) *ShellTranscriber {
	return &ShellTranscriber{cfg: cfg, run: defaultRun}
}

func defaultRun(ctx context.Context, name, _ string, args []string) error {
	// #nosec G204 -- name is an operator-configured tool; args are derived from
	// the audio path and config, not from untrusted network input.
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s: %w: %s", name, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// Transcribe runs the configured whisper CLI over audioPath and parses the
// WebVTT (or SRT) it produces.
func (s *ShellTranscriber) Transcribe(ctx context.Context, audioPath string) (*meeting.Transcript, error) {
	if _, err := os.Stat(audioPath); err != nil {
		return nil, fmt.Errorf("audio file: %w", err)
	}
	bin, err := s.resolveBinary()
	if err != nil {
		return nil, err
	}
	outDir, err := os.MkdirTemp("", "chiton-stt-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(outDir)

	args, err := s.buildArgs(bin, audioPath, outDir)
	if err != nil {
		return nil, err
	}
	if err := s.run(ctx, bin, outDir, args); err != nil {
		return nil, fmt.Errorf("transcription failed: %w", err)
	}

	subPath, err := firstSubtitle(outDir)
	if err != nil {
		return nil, fmt.Errorf("whisper produced no .vtt/.srt in %s: %w", outDir, err)
	}
	data, err := os.ReadFile(subPath)
	if err != nil {
		return nil, err
	}
	tr, err := meeting.Parse(filepath.Base(subPath), data)
	if err != nil {
		return nil, err
	}
	tr.Title = strings.TrimSuffix(filepath.Base(audioPath), filepath.Ext(audioPath))
	return tr, nil
}

// resolveBinary returns the configured binary, or the first auto-detected known
// whisper CLI on PATH.
func (s *ShellTranscriber) resolveBinary() (string, error) {
	if s.cfg.Binary != "" {
		return s.cfg.Binary, nil
	}
	for _, b := range knownBinaries {
		if _, err := exec.LookPath(b); err == nil {
			return b, nil
		}
	}
	return "", fmt.Errorf("no whisper CLI found on PATH (tried %s); install one, set --whisper-bin, "+
		"or build chiton with -tags whisper_cgo for the in-process backend", strings.Join(knownBinaries, ", "))
}

// buildArgs builds CLI args in the whisper.cpp form ("-m model -f audio -ovtt") or OpenAI's
// ("audio --output_format vtt"), chosen by the binary's base name.
func (s *ShellTranscriber) buildArgs(bin, audioPath, outDir string) ([]string, error) {
	base := strings.ToLower(filepath.Base(bin))
	base = strings.TrimSuffix(base, filepath.Ext(base))
	isCpp := base == "whisper-cli" || base == "whisper-cpp" || base == "whisper.cpp" || base == "main"

	if isCpp {
		if s.cfg.Model == "" {
			return nil, fmt.Errorf("whisper.cpp requires a ggml model path (set --model)")
		}
		args := []string{"-m", s.cfg.Model, "-f", audioPath, "-ovtt",
			"-of", filepath.Join(outDir, "out")}
		if s.cfg.Language != "" {
			args = append(args, "-l", s.cfg.Language)
		}
		return args, nil
	}

	// OpenAI whisper CLI.
	model := s.cfg.Model
	if model == "" {
		model = "base"
	}
	args := []string{audioPath, "--model", model, "--output_format", "vtt", "--output_dir", outDir}
	if s.cfg.Language != "" {
		args = append(args, "--language", s.cfg.Language)
	}
	return args, nil
}

// firstSubtitle finds the first .vtt (preferred) or .srt file in dir.
func firstSubtitle(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var srt string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		low := strings.ToLower(e.Name())
		if strings.HasSuffix(low, ".vtt") {
			return filepath.Join(dir, e.Name()), nil
		}
		if srt == "" && strings.HasSuffix(low, ".srt") {
			srt = filepath.Join(dir, e.Name())
		}
	}
	if srt != "" {
		return srt, nil
	}
	return "", fmt.Errorf("no subtitle file found")
}
