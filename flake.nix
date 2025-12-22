{
  description = "sttd - Speech-to-Text Daemon";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        # Python package for sttd
        sttd = python.pkgs.buildPythonApplication {
          pname = "sttd";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";

          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];

          propagatedBuildInputs = with python.pkgs; [
            faster-whisper
            sounddevice
            numpy
            click
            dasbus
            pillow
            pygobject3
            # Speaker identification dependencies
            scipy
            soundfile
            speechbrain
            huggingface-hub
            requests
          ];

          # Runtime dependencies
          makeWrapperArgs = [
            "--prefix" "LD_LIBRARY_PATH" ":" "${pkgs.portaudio}/lib"
            "--prefix" "PATH" ":" "${pkgs.lib.makeBinPath [ pkgs.wtype pkgs.wl-clipboard ]}"
            "--prefix" "GI_TYPELIB_PATH" ":" "${pkgs.glib.out}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0"
          ];

          # Skip tests during build
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Speech-to-Text Daemon for Wayland/Hyprland";
            homepage = "https://github.com/sdelcore/sttd";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };
      in
      {
        packages = {
          default = sttd;
          sttd = sttd;
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
            echo "sttd development environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"

            # Set up library paths for sounddevice and C++ stdlib
            export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

            # GI typelib path for PyGObject
            export GI_TYPELIB_PATH="${pkgs.glib}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"
          '';
        };
      }
    );
}
