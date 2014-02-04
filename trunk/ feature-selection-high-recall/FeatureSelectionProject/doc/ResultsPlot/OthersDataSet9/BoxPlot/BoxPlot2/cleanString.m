function [ S ] = cleanString( S )

 
 S = lower(regexprep(S,'[^a-zA-Z]',' '));

 skipset={'a','an','and','or','in','the'}; 


if iscell(S)
    S=cellfun(@(x) capitalize(x, skipset),S,'uniformoutput',false);
else
    skip=0;
    Sw=regexp(S,'[ \t]','split');
    skip(2:length(Sw))=ismember(lower(Sw(2:end)),skipset); %skip(1) is always 0
    Sw(~skip)=cellfun(@(x) [upper(x(1)) x(2:end)],Sw(~skip),'uniformoutput',false);
    S=sprintf('%s ',Sw{:});
    S(end)='';
    
    
end 
end

