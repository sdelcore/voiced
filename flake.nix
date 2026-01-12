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
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python with native extension dependencies
            (python.withPackages (ps: with ps; [
              # Native deps needed by pip packages
              numpy
              scipy
              av
              cryptography
              pyopenssl
              cffi
              # D-Bus/tray dependencies
              pygobject3
              pycairo
            ]))

            # uv for fast dependency management
            uv

            # C/C++ toolchain for native extensions
            stdenv.cc.cc.lib
            pkg-config

            # Audio
            portaudio
            ffmpeg

            # D-Bus and GLib for tray
            glib
            gobject-introspection
            cairo

            # Wayland tools
            wtype
            wl-clipboard

            # Development
            ruff
          ];

          shellHook = ''
            echo "voiced development environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"

            # Library paths
            export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

            # CUDA support
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              echo "CUDA: enabled"
            fi

            # GObject introspection
            export GI_TYPELIB_PATH="${pkgs.glib}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"

            # Create/sync venv with uv if needed
            if [ ! -d .venv ] || [ pyproject.toml -nt .venv ]; then
              echo "Syncing dependencies with uv..."
              uv sync --quiet
            fi

            # Activate venv
            source .venv/bin/activate
          '';
        };
      }
    );
}
