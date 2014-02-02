function output = acronym( input )
  words_cell = textscan(input,'%s','delimiter',' ');
  words      = words_cell{ : };
  letters    = cellfun(@(x) textscan(x,'%c%*s'), words);
  output     = upper(letters');
end