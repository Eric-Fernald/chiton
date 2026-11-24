package main

import (
	"fmt"
	"runtime"
	"runtime/debug"

	"github.com/spf13/cobra"
)

// These variables hold build metadata. Zero values mean a source build, and release builds inject real
// values via -ldflags "-X main.version=... -X main.commit=... -X main.date=...".
var (
	version = "dev"
	commit  = ""
	date    = ""
)

func versionCmd() *cobra.Command {
	var short bool
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print chiton's version and build metadata",
		RunE: func(cmd *cobra.Command, _ []string) error {
			out := cmd.OutOrStdout()
			v := resolveVersion()
			if short {
				fmt.Fprintln(out, v)
				return nil
			}
			fmt.Fprintf(out, "chiton %s\n", v)
			if commit != "" {
				fmt.Fprintf(out, "  commit: %s\n", commit)
			}
			if date != "" {
				fmt.Fprintf(out, "  built:  %s\n", date)
			}
			fmt.Fprintf(out, "  go:     %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
			return nil
		},
	}
	cmd.Flags().BoolVar(&short, "short", false, "Print just the version string")
	return cmd
}

// resolveVersion prefers the ldflags-injected version, then the module version or VCS revision
// from the embedded build info, so even source builds report more than a bare "dev".
func resolveVersion() string {
	if version != "dev" && version != "" {
		return version
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return version
	}
	if info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	var rev, modified string
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			modified = s.Value
		}
	}
	if rev != "" {
		if len(rev) > 12 {
			rev = rev[:12]
		}
		if modified == "true" {
			rev += "+dirty"
		}
		return "dev-" + rev
	}
	return version
}
