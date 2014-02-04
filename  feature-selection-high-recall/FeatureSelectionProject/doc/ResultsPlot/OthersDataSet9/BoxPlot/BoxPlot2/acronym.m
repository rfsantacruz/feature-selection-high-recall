function output = acronym( input )
  words_cell = textscan(input,'%s','delimiter',' ');
  words      = words_cell{ : };
  letters    = cellfun(@(x) textscan(x,'%c%*s'), words);
  if(length(letters) > 3)
    output = upper(letters(1:3)');
  else
     output = upper(letters');
end