{
  description = "voiced - Voice Daemon (STT + TTS)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        # Runtime dependencies
        runtimeDeps = with pkgs; [
          portaudio
          ffmpeg
          glib
          glib.dev
          gobject-introspection
          gobject-introspection.dev
          cairo
          cairo.dev
          wtype
          wl-clipboard
          stdenv.cc.cc.lib
          cacert
        ];

        # Python with base packages that have native deps
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          scipy
          av
          cryptography
          pyopenssl
          cffi
          pygobject3
          pycairo
        ]);

        # Source bundle for installation
        voicedSrc = pkgs.stdenv.mkDerivation {
          pname = "voiced-src";
          version = "0.2.0";
          src = ./.;
          phases = [ "installPhase" ];
          installPhase = ''
            mkdir -p $out
            cp -r $src/* $out/
          '';
        };

        # Wrapper script that manages venv and installs from bundled source
        voicedWrapper = pkgs.writeShellScriptBin "voiced" ''
          set -e

          VOICED_HOME="''${XDG_DATA_HOME:-$HOME/.local/share}/voiced"
          VENV_DIR="$VOICED_HOME/venv"
          VERSION_FILE="$VENV_DIR/.version"
          CURRENT_VERSION="0.2.0"
          SOURCE_DIR="${voicedSrc}"

          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeDeps}:''${LD_LIBRARY_PATH:-}"
          export GI_TYPELIB_PATH="${pkgs.glib}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0:''${GI_TYPELIB_PATH:-}"

          # Add CUDA support if available
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          fi

          # Check if venv needs to be created/updated
          if [ ! -f "$VERSION_FILE" ] || [ "$(cat "$VERSION_FILE")" != "$CURRENT_VERSION" ]; then
            echo "Setting up voiced v$CURRENT_VERSION..."
            mkdir -p "$VOICED_HOME"
            rm -rf "$VENV_DIR"

            ${pkgs.uv}/bin/uv venv "$VENV_DIR" --python ${pythonEnv}/bin/python --seed
            source "$VENV_DIR/bin/activate"

            # Install voiced from bundled source
            ${pkgs.uv}/bin/uv pip install "$SOURCE_DIR" --quiet

            echo "$CURRENT_VERSION" > "$VERSION_FILE"
            echo "Setup complete!"
          fi

          source "$VENV_DIR/bin/activate"
          exec "$VENV_DIR/bin/voiced" "$@"
        '';

        # FHS environment for full compatibility
        voicedFHS = pkgs.buildFHSEnv {
          name = "voiced";
          targetPkgs = pkgs: runtimeDeps ++ [
            pythonEnv
            pkgs.uv
            # Build tools needed for pycairo/pygobject compilation
            pkgs.gcc
            pkgs.pkg-config
            pkgs.meson
            pkgs.ninja
          ];
          runScript = "${voicedWrapper}/bin/voiced";
        };

      in
      {
        packages.default = voicedFHS;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            uv
            pkg-config
            portaudio
            ffmpeg
            glib
            gobject-introspection
            cairo
            wtype
            wl-clipboard
            ruff
          ];

          shellHook = ''
            echo "voiced development environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"

            export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              echo "CUDA: enabled"
            fi

            export GI_TYPELIB_PATH="${pkgs.glib}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"

            if [ ! -d .venv ] || [ pyproject.toml -nt .venv ]; then
              echo "Syncing dependencies with uv..."
              uv sync --quiet
            fi

            source .venv/bin/activate
          '';
        };
      }
    );
}
