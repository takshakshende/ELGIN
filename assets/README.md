# Assets

Place the following files here after running a training + rollout:

| File | How to generate | Used in README |
|---|---|---|
| `fluid_particles.gif` | Convert `fluid_speed_particles.mp4` to GIF (see note below) | Main animation |
| `compare.gif` | Convert `fluid_speed_compare.mp4` to GIF | Side-by-side comparison |
| `elgin_banner.png` | Optional project banner image | Header |

## Converting MP4 → GIF for GitHub README

GitHub READMEs display GIFs natively but not MP4.
Use ffmpeg to convert:

```bash
ffmpeg -i fluid_speed_particles.mp4 \
       -vf "fps=10,scale=750:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
       -loop 0 assets/fluid_particles.gif

ffmpeg -i fluid_speed_compare.mp4 \
       -vf "fps=10,scale=750:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
       -loop 0 assets/compare.gif
```

The generated MP4 files are located at:
`experiments/<case_name>/animations/fluid_speed_particles.mp4`
`experiments/<case_name>/animations/fluid_speed_compare.mp4`
