package config

import (
	"github.com/spf13/viper"
)

type Config struct {
	GraphPath string `mapstructure:"graph_path"`
	LogLevel  string `mapstructure:"log_level"`
}

// Load reads .chitonrc (TOML) from the current directory, merges environment
// overrides (CHITON_* prefix), and returns a Config with sane defaults.
func Load() (*Config, error) {
	v := viper.New()
	v.SetConfigName(".chitonrc")
	v.SetConfigType("toml")
	v.AddConfigPath(".")
	v.AddConfigPath("$HOME")

	v.SetDefault("graph_path", "./chiton-out/graph.kuzu")
	v.SetDefault("log_level", "info")

	v.SetEnvPrefix("CHITON")
	v.AutomaticEnv()

	// Config file is optional.
	_ = v.ReadInConfig()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
