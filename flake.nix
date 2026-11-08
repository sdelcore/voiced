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

        # Single source of truth: pyproject.toml.
        voicedVersion = (pkgs.lib.importTOML ./pyproject.toml).project.version;

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
          gtk3
          gtk3.dev
          gtk-layer-shell
          gtk-layer-shell.dev
          slurp
          wf-recorder
          pulseaudio
          pipewire
          # X11/XCB deps for pycairo build
          xorg.libxcb
          xorg.libxcb.dev
          xorg.libX11
          xorg.libX11.dev
          xorg.libXrender
          xorg.libXrender.dev
          xorg.xorgproto
          wtype
          wl-clipboard
          espeak-ng
          stdenv.cc.cc.lib
          cacert
          zlib
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
          version = voicedVersion;
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
          SOURCE_FILE="$VENV_DIR/.source"
          SOURCE_DIR="${voicedSrc}"

          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeDeps}:''${LD_LIBRARY_PATH:-}"
          export GI_TYPELIB_PATH="${pkgs.glib.out}/lib/girepository-1.0:${pkgs.gobject-introspection.out}/lib/girepository-1.0:${pkgs.gtk3}/lib/girepository-1.0:${pkgs.gtk-layer-shell}/lib/girepository-1.0:${pkgs.pango.out}/lib/girepository-1.0:${pkgs.gdk-pixbuf}/lib/girepository-1.0:${pkgs.atk}/lib/girepository-1.0:${pkgs.harfbuzz}/lib/girepository-1.0:''${GI_TYPELIB_PATH:-}"
          # Add pythonEnv site-packages to PYTHONPATH so gi/pycairo are available
          export PYTHONPATH="${pythonEnv}/${pythonEnv.sitePackages}:''${PYTHONPATH:-}"

          # Add CUDA support if available
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          fi

          # Avoid SSH config permission issues in FHS environment
          export GIT_SSH_COMMAND="ssh -F /dev/null"

          # Build the venv only when it is missing or its interpreter is gone,
          # which is the expensive path — it reinstalls the whole dependency
          # tree. A dangling bin/python (garbage-collected pythonEnv) fails the
          # -x test and rebuilds.
          if [ ! -x "$VENV_DIR/bin/python" ]; then
            echo "Creating voiced venv..."
            mkdir -p "$VOICED_HOME"
            rm -rf "$VENV_DIR"

            ${pkgs.uv}/bin/uv venv "$VENV_DIR" --python ${pythonEnv}/bin/python --seed --system-site-packages
          fi

          # Reinstall whenever the bundled source changes. The store path is
          # content addressed, so any edit gives a new one — keying on it
          # instead of the version means a source-only change still reaches the
          # venv. Keying on the version silently shipped stale code twice
          # (#13, #18). Deps are resolved here too, so adding a dependency
          # lands without a venv rebuild.
          # PyGObject removed from deps - provided by pythonEnv system-site-packages
          if [ ! -f "$SOURCE_FILE" ] || [ "$(cat "$SOURCE_FILE")" != "$SOURCE_DIR" ]; then
            echo "Installing voiced from $SOURCE_DIR..."
            ${pkgs.uv}/bin/uv pip install \
              --python "$VENV_DIR/bin/python" \
              --reinstall-package voiced \
              "$SOURCE_DIR" --quiet

            echo "$SOURCE_DIR" > "$SOURCE_FILE"
            rm -f "$VENV_DIR/.version"
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
            pkgs.git
            pkgs.openssh
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
            gtk3
            gtk-layer-shell
            slurp
            wf-recorder
            pulseaudio
            pipewire
            wtype
            wl-clipboard
            espeak-ng
            ruff
          ];

          shellHook = ''
            echo "voiced development environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"

            export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              echo "CUDA: enabled"
            fi

            export GI_TYPELIB_PATH="${pkgs.glib.out}/lib/girepository-1.0:${pkgs.gobject-introspection.out}/lib/girepository-1.0:${pkgs.gtk3}/lib/girepository-1.0:${pkgs.gtk-layer-shell}/lib/girepository-1.0:${pkgs.pango.out}/lib/girepository-1.0:${pkgs.gdk-pixbuf}/lib/girepository-1.0:${pkgs.atk}/lib/girepository-1.0:${pkgs.harfbuzz}/lib/girepository-1.0:$GI_TYPELIB_PATH"

            if [ ! -d .venv ] || [ pyproject.toml -nt .venv ]; then
              echo "Syncing dependencies with uv..."
              uv sync --quiet
            fi

            source .venv/bin/activate
          '';
        };
      }
    ) // {
      # Outside eachDefaultSystem on purpose: a home-manager module is
      # system-independent and resolves the package from the importing
      # config's own pkgs.
      homeModules.default = import ./nix/hm-module.nix { inherit self; };
      homeModules.voiced = self.homeModules.default;
    };
}
