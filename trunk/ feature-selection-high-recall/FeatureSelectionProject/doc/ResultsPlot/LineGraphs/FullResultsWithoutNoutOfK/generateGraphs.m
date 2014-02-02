%script to generate graphs
%todo: specify the labels for each algoritghm, it follows the order in
%wich the algorithms appears in the xml

%clear 
clear all, close all;

data = xml2struct('featureSelection.xml');

%config legends and plot colors => TODO
%legendKeys = data.simulation.Attributes.algs;
%legendDef = containers.Map(legendKeys, {'',''});

%loop over Data Sets
nDS = size(data.Simulation.DataSet,2);
for d=1:nDS
    dSName = cleanString(data.Simulation.DataSet{d}.Attributes.name);
    dSNumFea = str2double(data.Simulation.DataSet{d}.Attributes.nfeatures);
    
    %loop over classifiers
    nCl = size(data.Simulation.DataSet{d}.Classifier,2);
    for c = 1:nCl
        CName = cleanString(data.Simulation.DataSet{d}.Classifier{c}.Attributes.name);
        
        %loop over metrics
        nM = size(data.Simulation.DataSet{d}.Classifier{c}.Metric,2);
        for m = 1:nM
            MName = cleanString(data.Simulation.DataSet{d}.Classifier{c}.Metric{m}.Attributes.name);
            nAlg = size(data.Simulation.DataSet{d}.Classifier{c}.Metric{m}.Algorithms.Algorithm,2);
                               
            %plot each graph
            figure, hold on;
            
            %TODO: define symbols 
            plotSymbols = {'-g<','-mv','-mo','-m^','-gs','-g>','-mx','-mh','-bp','-bd','-rp','-rd'};
            legendKeys = {}; 
            for alg = 1:nAlg
                mean = str2num(data.Simulation.DataSet{d}.Classifier{c}.Metric{m}.Algorithms.Algorithm{alg}.Mean.Text);
                algname = cleanString(data.Simulation.DataSet{d}.Classifier{c}.Metric{m}.Algorithms.Algorithm{alg}.Attributes.name);
                legendKeys = [legendKeys, algname];
                plot(1:dSNumFea, mean, plotSymbols{alg})
                
                
            end;
            
            %plot settings
            title(cleanString(strjoin({dSName, CName, MName},' ')));
            xlabel('Number of features');
            ylabel(MName);
            legend(legendKeys,'Location', 'SouthOutside');
            %axis([1 dSNumFea 0 1])
            saveas(gcf, strjoin({dSName, CName, MName},' ') , 'pdf')
            hold off;
            
        end;
    end;
end;