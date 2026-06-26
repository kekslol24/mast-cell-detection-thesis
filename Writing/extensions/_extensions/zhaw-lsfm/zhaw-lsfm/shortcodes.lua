-- Shortcodes for generating figure and table lists in Typst
return {
  ['list-of-figures'] = function(args, kwargs, meta)
    -- Generate typst code for figure list
    local typst_code = '#outline(title: none, target: figure.where(kind: "quarto-float-fig"))'
    return pandoc.RawBlock('typst', typst_code)
  end,
  
  ['list-of-tables'] = function(args, kwargs, meta)
    -- Generate typst code for table list  
    local typst_code = '#outline(title: none, target: figure.where(kind: "quarto-float-tbl"))'
    return pandoc.RawBlock('typst', typst_code)
  end,
  
  ['table-of-contents'] = function(args, kwargs, meta)
    -- Generate typst code for table of contents with depth awareness
    local toc_depth = 2 -- default depth to match ZHAW requirements
    
    -- Check if toc-depth is specified in metadata
    if meta["toc-depth"] then
      toc_depth = tonumber(pandoc.utils.stringify(meta["toc-depth"])) or 2
    end
    
    local typst_code = '#outline(title: none, depth: ' .. toc_depth .. ')'
    return pandoc.RawBlock('typst', typst_code)
  end,
  
  ['references'] = function(args, kwargs, meta)
    -- Generate references div for bibliography
    return pandoc.Div({}, pandoc.Attr("refs"))
  end
}