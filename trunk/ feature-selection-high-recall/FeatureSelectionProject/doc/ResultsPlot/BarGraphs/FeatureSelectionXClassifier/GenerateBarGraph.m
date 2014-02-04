%script to generate barGraph, major = feature selection, minor = classifier

%clear
clear all, close all;

%xml file
file = 'featureSelection.xml';

%read the xml to struct
data = xml2struct(file);

%number of data sets
nDS = size(data.Simulation.DataSet,2);
for d=1:nDS
    dSName = data.Simulation.DataSet{d}.Attributes.name;
    
    %loop over metrics
    nM = size(data.Simulation.DataSet{d}.Classifier{1}.Metric,2);
    for m = 1:nM
        MName = data.Simulation.DataSet{d}.Classifier{1}.Metric{m}.Attributes.name;
        nCl = size(data.Simulation.DataSet{d}.Classifier,2);
        nAlg = size(data.Simulation.DataSet{d}.Classifier{1}.Metric{m}.Algorithms.Algorithm,2);
        Y = zeros(nAlg,nCl);
        Error = zeros(nAlg,nCl);
        
        algLegend = {};
        for alg = 1:nAlg
            algname = data.Simulation.DataSet{d}.Classifier{1}.Metric{m}.Algorithms.Algorithm{alg}.Attributes.name;
            algLegend = [algLegend, cleanString(algname)];
            classifiersLegends = {};
            %loop over classifiers
            nCl = size(data.Simulation.DataSet{1}.Classifier,2);
            for c = 1:nCl
                CName = data.Simulation.DataSet{1}.Classifier{c}.Attributes.name;
                classifiersLegends = [classifiersLegends, cleanString(CName)];
                
                %xpath expression
                exp = strcat('//DataSet[@name="',dSName ,'"]/Classifier[@name="',CName,'"]/Metric[@name="',MName,'"]/Algorithms/Algorithm[@name="',algname,'"]/Mean');
                nodeList = queryXml(exp,file);
                
                % Iterate through the nodes that are returned. but it is just
                % one node in each interation
                acc = 0.0;
                count = 0;
                samples = [];
                for i = 1:nodeList.getLength
                    node = nodeList.item(i-1);
                    %convert text matrix to number matrix
                    values = str2num( node.getFirstChild.getNodeValue);
                    samples = [samples, values];
                    acc = acc + sum(values);
                    count = count + length(values);
                    %disp(str2num( node.getFirstChild.getNodeValue))
                end;
                Y(alg,c) = acc / count;
                Error(alg,c) = 2 * (std(samples)/sqrt(length(samples)));
            end;
            Y(alg,:) = Y(alg,:) ./ max(Y(alg,:));
        end;
        
        %use just initials
        for i=1:length(algLegend)
            algLegend{i} = strjoin(acronym(algLegend{i}),'');
        end;
        
        %plot graph
        %createBarGraph(Y,algLegend,classifiersLegends, cleanString(strjoin({dSName,MName},' ')), 'Average', 'Feature selection algorithm')
        
        figure, barweb(Y, Error, 1, algLegend, cleanString(strjoin({dSName,MName},' ')), 'Feature Selection Algorithms', 'Average', [], [],classifiersLegends , 2, 'plot')
        %save as pdf
        saveas(gcf, strjoin({dSName,MName},' '), 'pdf')
        
    end;
    
end;

