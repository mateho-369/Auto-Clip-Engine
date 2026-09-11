# Bundled Khmer fonts (SIL Open Font License 1.1)

All caption/UI fonts ship here so rendering never depends on system fonts or a CDN.

| File | Family | License |
|---|---|---|
| NotoSansKhmer-*.ttf | Noto Sans Khmer | OFL-NotoSansKhmer.txt (SIL OFL 1.1) |
| NotoSerifKhmer_*.ttf | Noto Serif Khmer | OFL-NotoSerifKhmer.txt (SIL OFL 1.1) |
| Battambang-*.ttf | Battambang | OFL-Battambang.txt (SIL OFL 1.1) |
| KantumruyPro_*.ttf | Kantumruy Pro | OFL-KantumruyPro.txt (SIL OFL 1.1) |
| Moul-Regular.ttf | Moul | OFL-Moul.txt (SIL OFL 1.1) |

OFL 1.1 permits bundling + redistribution + commercial use in video output.
Reserved Font Names apply to the fonts themselves (renaming/subsetting rules);
rendering captions with them in exported MP4s is normal permitted use.

Python-side deps used by this feature: `uharfbuzz` (shaped width measurement,
same HarfBuzz libass uses) and `khmercut` (offline Khmer dictionary word
segmentation). Both are optional at runtime with safe fallbacks; install for
exact wrapping:  pip install uharfbuzz khmercut
