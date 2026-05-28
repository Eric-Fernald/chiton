package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const (
	defaultOllamaEndpoint = "http://localhost:11434"
	defaultOllamaModel    = "nomic-embed-text"
	defaultOllamaDim      = 768
)

// Ollama is a thin client over a local Ollama server's /api/embed endpoint.
// It pulls in no SDK — just net/http — to keep chiton a single static binary.
type Ollama struct {
	endpoint string
	model    string
	dim      int
	client   *http.Client
}

// NewOllama builds an Ollama embedder from cfg, applying defaults for any
// empty/zero field (endpoint http://localhost:11434, model nomic-embed-text,
// dim 768).
func NewOllama(cfg Config) *Ollama {
	endpoint := strings.TrimRight(cfg.Endpoint, "/")
	if endpoint == "" {
		endpoint = defaultOllamaEndpoint
	}
	model := cfg.Model
	if model == "" {
		model = defaultOllamaModel
	}
	dim := cfg.Dim
	if dim <= 0 {
		dim = defaultOllamaDim
	}
	return &Ollama{
		endpoint: endpoint,
		model:    model,
		dim:      dim,
		client:   &http.Client{Timeout: 60 * time.Second},
	}
}

func (o *Ollama) Dim() int      { return o.dim }
func (o *Ollama) Model() string { return o.model }

type ollamaRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Error      string      `json:"error,omitempty"`
}

// Embed calls Ollama's batch embedding endpoint once for all texts.
func (o *Ollama) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(ollamaRequest{Model: o.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := o.endpoint + "/api/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: POST %s failed (is `ollama serve` running?): %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var er ollamaResponse
		_ = json.NewDecoder(resp.Body).Decode(&er)
		if er.Error != "" {
			return nil, fmt.Errorf("ollama: %s (status %d) — try `ollama pull %s`", er.Error, resp.StatusCode, o.model)
		}
		return nil, fmt.Errorf("ollama: unexpected status %d from %s", resp.StatusCode, url)
	}

	var out ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	if out.Error != "" {
		return nil, fmt.Errorf("ollama: %s", out.Error)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama: expected %d embeddings, got %d", len(texts), len(out.Embeddings))
	}

	// Trust the model's true dimension over the configured default; warn-free
	// self-correction keeps the index consistent if dim was misconfigured.
	if len(out.Embeddings[0]) > 0 {
		o.dim = len(out.Embeddings[0])
	}
	return out.Embeddings, nil
}
