{
  description = "Artlite's development shell for NixOS: the build's tools and libdrm; uv installs the rest";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "artlite";
        packages = [ pkgs.uv pkgs.cmake pkgs.pkg-config ];
        buildInputs = [ pkgs.libdrm ];
        # The Python uv installs and the toolchain wheels' programs (Peano's clang, aie-opt, aiecc) are built for other
        # distributions: NixOS runs them through nix-ld (programs.nix-ld.enable = true), here on this shell's glibc
        # (the native module's) with the libraries the wheels link beyond it. uv's own Python only: a Nix-built one
        # would load the wheels' modules without nix-ld's libraries.
        env = {
          NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
          UV_PYTHON_PREFERENCE = "only-managed";
        };
      };
    };
}
