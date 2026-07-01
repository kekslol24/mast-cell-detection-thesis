// This is an example typst template (based on the default template that ships
// with Quarto). It defines a typst function named 'article' which provides
// various customization options. This function is called from the 
// 'typst-show.typ' file (which maps Pandoc metadata function arguments)
//
// If you are creating or packaging a custom typst template you will likely
// want to replace this file and 'typst-show.typ' entirely. You can find 
// documentation on creating typst templates and some examples here: 
//   - https://typst.app/docs/tutorial/making-a-template/
//   - https://github.com/typst/templates


#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  institut: none,
  institute: none,
  keywords: none,
  confidential: false,
  thesis-type: none,
  degree-type: none,
  study-year: none,
  submission-date: none,
  study-direction: none,
  supervisor: none,
  co-examiner: none,
  cover-image: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 2cm),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: "Helvetica",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "libertinus serif",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  number-depth: none,
  pagenumbering: "1",
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: none,
  )
  set par(leading: 1.0em, justify: true)
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(
    numbering: if sectionnumbering != none {
      (..nums) => if nums.pos().len() <= number-depth {
        numbering(sectionnumbering, ..nums)
      }
    } else { none }
  )
  // State to track if we're in appendix mode
  let appendix-mode = state("appendix-mode", false)
  
  show heading: it => {
    it
    v(0.5em)
  }
  // Pagebreaks are handled by Lua filter to avoid container conflicts
  
  // Title page
  if title != none or authors != none {
    page(numbering: none)[
      #align(center)[
        #if lang == "en" [
          ZURICH UNIVERSITY OF APPLIED SCIENCES \
          DEPARTEMENT LIFE SCIENCES AND FACILITY MANAGEMENT
        ] else [
          ZÜRCHER HOCHSCHULE FÜR ANGEWANDTE WISSENSCHAFTEN \
          DEPARTEMENT LIFE SCIENCES UND FACILITY MANAGEMENT
        ]
        #if institut != none [
          \ #upper(institut)
        ]
      ]
      #v(1fr)
      #if title != none {
        align(center)[#block(inset: 2em)[
          #set par(leading: heading-line-height)
          #text(weight: "bold", size: title-size)[#title]
          #if subtitle != none {
            parbreak()
            text(weight: "bold", size: subtitle-size)[#subtitle]
          }
          #if cover-image != none {
            v(0.5cm)
            image(cover-image.src, width: eval(cover-image.max-width), height: eval(cover-image.max-height), fit: "contain")
            v(0.1cm)
          }
          #if confidential {
            parbreak()
            if lang == "en" [
              #text(weight: "bold", size: subtitle-size)[Confidential]
            ] else [
              #text(weight: "bold", size: subtitle-size)[Vertraulich]
            ]
          }
          #if thesis-type != none {
            v(0.2cm)
            text(weight: "bold", size: subtitle-size)[#thesis-type]
          }
        ]]
      }

      #v(0.5cm)

      #if authors != none {
        align(center)[
          #if lang == "en" [
            by \
          ] else [
            von \
          ]
          #for author in authors [
            #author.name \
          ]
          #if degree-type != none and study-year != none [
            \ #degree-type #study-year
          ]
          #if submission-date != none [
            \ #if lang == "en" [
              Submission date: #submission-date
            ] else [
              Abgabedatum: #submission-date
            ]
          ]
          #if study-direction != none [
            \ #if lang == "en" [
              Study Direction: #study-direction
            ] else [
              Studiengang: #study-direction
            ]
          ]
        ]
      }

      #v(10fr)
      
      #if supervisor != none {
        align(left)[
          #if lang == "en" [
            *Supervisor:* \
          ] else [
            *Betreuer / Betreuerinnen:* \
          ]
          #for supervisor in supervisor [
            #if supervisor.title != none [#supervisor.title] #supervisor.name \
            #supervisor.affiliation \
            \
          ]
        ]
      }

      #v(1fr)

      #if co-examiner != none {
        align(left)[
          #if lang == "en" [
            *Co-Examiner:* \
          ] else [
            *Zweitkorrektur:* \
          ]
          #for corrector in co-examiner [
            #if corrector.title != none [#corrector.title] #corrector.name \
            #corrector.affiliation \
            \
          ]
        ]
      }
    ]
  }

// Imprint page (second page)
  if title != none or authors != none {
    page(numbering: none)[
      #v(1fr)
      
      #align(left)[
        #if lang == "en" [
          *Imprint*
        ] else [
          *Impressum*
        ]
      ]
      
      #v(2em)
      
      // 1. Institute section
      #if institute != none {
        align(left)[
          #if lang == "en" [
            *Institute:* \
            #institute
          ] else [
            *Institut:* \
            #institute
          ]
        ]
        v(2em)
      } else if institut != none {
        align(left)[
          #if lang == "en" [
            *Institute:* \
            #institut
          ] else [
            *Institut:* \
            #institut
          ]
        ]
        v(2em)
      }
      
      // 2. Citation section
      #if authors != none and title != none {
        align(left)[
          #if lang == "en" [
            *Recommended Citation:*
          ] else [
            *Zitiervorschlag:*
          ]
        ]
        
        align(left)[
          #text(size: 10pt)[
            #authors.map(author => author.name).join(", ")#if date != none [ (#date).] else if submission-date != none [ (#{let date-str = repr(submission-date).trim("[").trim("]"); date-str.split("-").at(0)}).] else [ (n.d.).] _#title#if subtitle != none [: #subtitle]_#if institute != none [, #institute.]
          ]
        ]
        
        v(2em)
      }
      
      // 3. Keywords
      #if keywords != none {
        align(left)[
          #if lang == "en" [
            *Keywords:* #keywords
          ] else [
            *Schlagworte:* #keywords
          ]
        ]
      }
    ]
  }

  // Start page numbering for main content with header
  set page(
    numbering: pagenumbering,
    header: [
      #text(size: 0.8em)[
        #grid(
          columns: (1fr, 1fr, 1fr),
          align: (left, center, right),
          gutter: 1em,
          [ZHAW LSFM],
          [#if thesis-type != none [#thesis-type]],
          [#if authors != none [
            #for author in authors [
              #author.name
              #if author != authors.last() [, ]
            ]
          ]]
        )
      ]
      #v(0.3em)
      #line(length: 100%, stroke: 0.5pt)
    ],
    footer: [
      #text(size: 0.8em)[
        #align(center)[
          #context counter(page).display()
        ]
      ]
    ]
  )
  counter(page).update(1)

  set block(spacing: 2em)
  set par(leading: 0.65em)

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: (x, y) => if y == 0 { (bottom: 0.6pt) },
  fill: (_, row) => if row == 0 {
    rgb("#d9d9d9")
  } else if calc.odd(row) {
    white
  } else {
    rgb("#f2f2f2")
  },
)
