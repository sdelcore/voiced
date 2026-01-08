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

        # Python package for voiced
        voiced = python.pkgs.buildPythonApplication {
          pname = "voiced";
          version = "0.2.0";
          src = ./.;
          format = "pyproject";

          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];

          propagatedBuildInputs = with python.pkgs; [
            # STT dependencies
            faster-whisper
            speechbrain
            # TTS dependencies
            tqdm
            websockets
            # Shared dependencies
            sounddevice
            soundfile
            numpy
            scipy
            click
            requests
            huggingface-hub
            # Linux/Wayland integration
            dasbus
            pillow
            pygobject3
            # Speaker identification
            scikit-learn
          ];

          # Runtime dependencies
          makeWrapperArgs = [
            "--prefix" "LD_LIBRARY_PATH" ":" "${pkgs.portaudio}/lib"
            "--prefix" "PATH" ":" "${pkgs.lib.makeBinPath [ pkgs.wtype pkgs.wl-clipboard pkgs.ffmpeg ]}"
            "--prefix" "GI_TYPELIB_PATH" ":" "${pkgs.glib.out}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0"
          ];

          # Skip tests during build
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Voice Daemon - STT and TTS for Wayland/Hyprland";
            homepage = "https://github.com/sdelcore/voiced";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };
      in
      {
        packages = {
          default = voiced;
          voiced = voiced;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python with all dependencies pre-installed from Nix
            # This ensures C extensions are properly linked
            (python312.withPackages (ps: with ps; [
              # Core dependencies
              faster-whisper
              sounddevice
              numpy
              click
              pillow
              tqdm
              websockets
              # D-Bus/tray dependencies
              pygobject3
              pycairo
              dasbus
            ]))
            uv

            # C++ standard library for any remaining pip packages
            stdenv.cc.cc.lib

            # Audio backend for sounddevice
            portaudio

            # D-Bus and GLib for StatusNotifierItem tray
            glib
            gobject-introspection
            cairo
            pkg-config

            # Wayland text input
            wtype
            wl-clipboard

            # Audio format handling
            ffmpeg

            # Development tools
            ruff
          ];

          shellHook = ''
            echo "voiced development environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"

            # Set up library paths for sounddevice and C++ stdlib
            export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

            # Add NVIDIA driver libs for CUDA/GPU support
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              echo "CUDA: enabled (found /run/opengl-driver/lib)"
            fi

            # GI typelib path for PyGObject
            export GI_TYPELIB_PATH="${pkgs.glib}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"
          '';
        };
      }
    );
}
