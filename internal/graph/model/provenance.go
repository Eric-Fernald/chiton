package model

// DefaultConfidenceThreshold splits INFERRED from AMBIGUOUS when none is set.
const DefaultConfidenceThreshold float32 = 0.5

// ResolveProvenance maps a host-supplied confidence to a provenance tag and a
// clamped confidence. At or above threshold → INFERRED; below → AMBIGUOUS.
func ResolveProvenance(confidence, threshold float32) (Provenance, float32) {
	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}
	if confidence >= threshold {
		return ProvInferred, confidence
	}
	return ProvAmbiguous, confidence
}

// AttrString reads a string-valued attr, returning "" when attrs is nil, the
// key is absent, or the value is not a string.
func AttrString(attrs map[string]any, key string) string {
	if attrs == nil {
		return ""
	}
	if v, ok := attrs[key].(string); ok {
		return v
	}
	return ""
}

// AttrStrings reads a []string-ish attr, tolerating both []string and []any
// (the latter occurs after a JSON round-trip through the store).
func AttrStrings(attrs map[string]any, key string) []string {
	if attrs == nil {
		return nil
	}
	switch v := attrs[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, x := range v {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
