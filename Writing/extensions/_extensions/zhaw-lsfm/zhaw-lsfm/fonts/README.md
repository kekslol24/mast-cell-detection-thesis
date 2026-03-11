# Fonts Directory

This directory can contain local font files (.ttf or .otf) that will be used by Typst in addition to system fonts.

## For Arial Font Support

To ensure consistent Arial rendering across all systems, you can place Arial font files here:

- `arial.ttf` - Arial Regular
- `arialbd.ttf` - Arial Bold  
- `ariali.ttf` - Arial Italic
- `arialbi.ttf` - Arial Bold Italic

## How Font Fallbacks Work

The template uses this font fallback chain:

1. **Arial** - Preferred font (ZHAW requirement)
2. **Helvetica Neue** - macOS system font
3. **Helvetica** - Cross-platform fallback
4. **Liberation Sans** - Linux open-source equivalent
5. **DejaVu Sans** - Widely available fallback
6. **Noto Sans** - Google's universal fallback font

## Usage

Simply place your font files in this directory. Typst will automatically find and use them if they match the font names in the template.