package url

import (
	"bytes"
	"io"
	"strings"

	"github.com/ledongthuc/pdf"
	"golang.org/x/net/html"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
)

// extractText renders a fetched payload to plain text by content type (falling back to extension),
// plus the document title when the format has one (HTML <title>).
func extractText(fr *fetchResult) (text, title string) {
	ct := fr.contentType
	switch {
	case strings.Contains(ct, "pdf"), fr.ext == "pdf":
		return pdfText(fr.body), ""
	case strings.Contains(ct, "html"), fr.ext == "html", fr.ext == "htm":
		return htmlText(fr.body)
	case strings.Contains(ct, "vtt"), fr.ext == "vtt",
		strings.Contains(ct, "subrip"), fr.ext == "srt":
		return transcriptText(fr), ""
	default:
		return string(fr.body), ""
	}
}

// htmlText extracts visible text and the <title> from an HTML document, skipping
// <script>, <style>, <head> noise, and collapsing whitespace per block element.
func htmlText(body []byte) (text, title string) {
	doc, err := html.Parse(bytes.NewReader(body))
	if err != nil {
		// Fall back to a crude tag strip if the parser chokes.
		return stripTags(string(body)), ""
	}
	var b strings.Builder
	var walk func(*html.Node, bool)
	walk = func(n *html.Node, inBody bool) {
		switch n.Type {
		case html.ElementNode:
			switch n.Data {
			case "script", "style", "noscript", "template", "svg":
				return
			case "title":
				if title == "" {
					title = strings.TrimSpace(textContent(n))
				}
				return
			case "body":
				inBody = true
			case "br":
				b.WriteByte('\n')
			case "p", "div", "section", "article", "li", "tr", "h1", "h2",
				"h3", "h4", "h5", "h6", "header", "footer", "blockquote", "pre":
				defer b.WriteString("\n\n")
			}
		case html.TextNode:
			if inBody {
				t := strings.TrimSpace(n.Data)
				if t != "" {
					b.WriteString(t)
					b.WriteByte(' ')
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c, inBody)
		}
	}
	walk(doc, false)
	return collapseBlankLines(b.String()), title
}

// textContent returns the concatenated text of a node's subtree.
func textContent(n *html.Node) string {
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			b.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return b.String()
}

// pdfText extracts plain text from a PDF. ledongthuc/pdf can panic on malformed input, so a recover
// turns any failure into "" (Add then reports no readable text).
func pdfText(body []byte) (out string) {
	defer func() {
		if r := recover(); r != nil {
			out = ""
		}
	}()
	r, err := pdf.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return ""
	}
	rd, err := r.GetPlainText()
	if err != nil {
		return ""
	}
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, rd); err != nil {
		return ""
	}
	return buf.String()
}

// transcriptText renders a VTT/SRT file to prose via the meeting parser, dropping timing markup that
// would pollute concept extraction.
func transcriptText(fr *fetchResult) string {
	name := "subtitles." + fr.ext
	tr, err := meeting.Parse(name, fr.body)
	if err != nil {
		return string(fr.body)
	}
	return tr.Text()
}

// stripTags is a last-resort HTML tag remover for documents the parser rejects.
func stripTags(s string) string {
	var b strings.Builder
	depth := 0
	for _, r := range s {
		switch r {
		case '<':
			depth++
		case '>':
			if depth > 0 {
				depth--
			}
		default:
			if depth == 0 {
				b.WriteRune(r)
			}
		}
	}
	return collapseBlankLines(b.String())
}

// collapseBlankLines trims trailing spaces and collapses runs of blank lines to
// a single blank line, so chunkText's paragraph splitting stays meaningful.
func collapseBlankLines(s string) string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	blank := 0
	for _, ln := range lines {
		ln = strings.TrimRight(ln, " \t")
		if strings.TrimSpace(ln) == "" {
			blank++
			if blank > 1 {
				continue
			}
			out = append(out, "")
			continue
		}
		blank = 0
		out = append(out, strings.Join(strings.Fields(ln), " "))
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}
