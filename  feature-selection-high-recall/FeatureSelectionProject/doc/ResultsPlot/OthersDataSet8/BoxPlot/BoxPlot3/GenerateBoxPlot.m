%script to generate barGraph

%clear
clear all, close all;

%xml file
file = 'featureSelection.xml';

%read the xml to struct
data = xml2struct(file);

%number of data sets
nDS = size(data.Simulation.DataSet,2);

%number of algs
nAlg = size(data.Simulation.DataSet{1}.Classifier{1}.Metric{1}.Algorithms.Algorithm,2);


%loop over metrics
nM = size(data.Simulation.DataSet{1}.Classifier{1}.Metric,2);
for m = 1:nM
    MName = data.Simulation.DataSet{1}.Classifier{1}.Metric{m}.Attributes.name;
    
    %loop over classifiers
    nCl = size(data.Simulation.DataSet{1}.Classifier,2);
    for c = 1:nCl
        CName = data.Simulation.DataSet{1}.Classifier{c}.Attributes.name;
        
        %compute average and plot
        Y = zeros(nDS,nAlg);
        dataSetLegends = {};
        for d=1:nDS
            dSName = data.Simulation.DataSet{d}.Attributes.name;
            dataSetLegends = [dataSetLegends, cleanString(dSName)];
            legendKeys = {};
            
            for alg = 1:nAlg
                algname = data.Simulation.DataSet{d}.Classifier{1}.Metric{m}.Algorithms.Algorithm{alg}.Attributes.name;
                legendKeys = [legendKeys, cleanString(algname)];
                
                %xpath expression
                exp = strcat('//DataSet[@name="',dSName ,'"]/Classifier[@name="',CName,'"]/Metric[@name="',MName,'"]/Algorithms/Algorithm[@name="',algname,'"]/Mean');
                
                acc = 0.0;
                count = 0;
                nodeList = queryXml(exp,file);
                for i = 1:nodeList.getLength
                    node = nodeList.item(i-1);
                    %convert text matrix to number matrix
                    values = str2num( node.getFirstChild.getNodeValue);
                    acc = acc + sum(values);
                    count = count + length(values);
                    %disp(str2num( node.getFirstChild.getNodeValue))
                end;
                Y(d,alg) = acc/count;
            end;
            
            %normalize
            Y(d,:) = Y(d,:) ./ max(Y(d,:));
        end;
        
        %plot graph
        createBoxPlot(Y,legendKeys, cleanString(strjoin({MName,CName}, ' ')), 'Average', 'Feature Selection Algorithm')
        
        %save as pdf
        saveas(gcf, cleanString(strjoin({MName,CName}, ' ')) , 'pdf')
    end;
    
end;




