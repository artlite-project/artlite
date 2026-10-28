{
  description = "Artlite dev shell (uv + NixOS)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    nix-amd-ai.url = "github:noamsto/nix-amd-ai";
  };

  outputs = { self, nixpkgs, nix-amd-ai }:
    let
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          amd-ai = nix-amd-ai.packages.${system};
        in
        {
          default = pkgs.mkShell {
            name = "artlite";

            # Tools required to configure and build the C++ extensions.
            nativeBuildInputs = with pkgs; [
              uv
              python312
              cmake
              pkg-config
            ];

            # System libraries the amdxdna shim links against, plus the GCC C++
            # runtime that binary Python wheels need at import time.
            buildInputs = with pkgs; [
              libdrm
              libuuid
              stdenv.cc.cc.lib
              amd-ai.xrt
            ];

            # NixOS' GCC wrapper enables -Werror=format-security by default.
            # csrc/artlite/profiler.cpp passes a runtime format string to fprintf,
            # so disable that hardening for the dev shell rather than patching upstream.
            hardeningDisable = [ "format" ];

            shellHook = ''
              # Build both the amdxdna shim and the optional XRT backend.
              export CMAKE_ARGS="-DARTLITE_XRT=ON"

              # Point Artlite's CMake and the XRT runtime to the Nix AMD AI package.
              export XILINX_XRT="${amd-ai.xrt}"

              # Make uv use the Python we provide from nixpkgs.
              export UV_PYTHON=3.12
              export UV_PYTHON_PREFERENCE=only-system

              # Put the GCC C++ standard library on the runtime loader path so
              # wheels compiled against libstdc++ (NumPy, torch, etc.) load it.
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              # Expose the XRT runtime libraries to the loader for artlite._native_xrt.
              export LD_LIBRARY_PATH="${amd-ai.xrt}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              # Activate the uv-managed venv if it already exists so plain `python`
              # works the same as `uv run python`.
              if [ -f .venv/bin/activate ]; then
                source .venv/bin/activate
              fi
            '';
          };
        });
    };
}
