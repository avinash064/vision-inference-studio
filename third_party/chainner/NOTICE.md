# chaiNNer backend (modified snapshot)

`src/` is a modified snapshot of the Python processing backend of
[chaiNNer](https://github.com/chaiNNer-org/chaiNNer) (`backend/src` upstream), licensed under the
GNU General Public License v3.0 (see `LICENSE`). It has been adapted to run inside Vision Inference Studio,
which imports its node registry and executes its nodes in memory.

Vision Inference Studio ships this code in its desktop packages (`resources/chainner/src`); this directory is
the corresponding source. `src/texconv/texconv.exe` is Microsoft's DirectXTex texconv, under the MIT license in
`src/texconv/LICENSE`.
