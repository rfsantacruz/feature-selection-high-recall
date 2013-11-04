package featureSelection;

import classifiers.NaiveBayesClassifier;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class FeatureSelectionFilterFactory {

	//singleton impl
	private static FeatureSelectionFilterFactory instance;
	public synchronized static FeatureSelectionFilterFactory getInstance(){
		if(instance == null)
			instance = new FeatureSelectionFilterFactory();

		return instance;
	}
	public FeatureSelectionFilterFactory(){}

	//create filter to be used in the simulations 
	public Filter createFilter(EFeatureSelectionAlgorithm EType,FeatureSelectionFactoryParameters parameter) throws Exception{
		
		//base classes
		AttributeSelection attributeSelection = new AttributeSelection();
		ASEvaluation evaluator = null;
		ASSearch search = null;
		
		//set the evaluator and searcher for each algorithm
		switch (EType) {
		case INFORMATIONGAIN_RANK:
			//mutual information evaluator
			evaluator = new InfoGainAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;
			
		case CONDITIONAL_ENTROPY_RANK:
			//conditional entropy evaluator
			evaluator = new ConditionalEntropyFeatureSelection();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;
			
		case GAINRATIO_RANK:
			//conditional entropy evaluator
			evaluator = new GainRatioAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;
			
		case SYMMETRICAL_UNCERT_RANK:
			//conditional entropy evaluator
			evaluator = new SymmetricalUncertAttributeEval();
			//rank the features based on the evaluator
			search = new Ranker();
			((Ranker) search).setNumToSelect(parameter.getNumberOfFeature());
			break;
			
		case CORRELATION_BASED_SUBSET:
			//correlation based evaluator
			evaluator = new CfsSubsetEval();
			//choose based on a graddy search 
			search = new GreedyStepwise();
			((GreedyStepwise)search).setGenerateRanking(true);
			((GreedyStepwise)search).setNumToSelect(parameter.getNumberOfFeature());
			break;
			
		case MRMR_MI_BASED_SUBSET:
			//correlation based evaluator
			evaluator = new MRMRFeatureSelection();
			//choose based on a graddy search 
			search = new GreedyStepwise();
			((GreedyStepwise)search).setGenerateRanking(true);
			((GreedyStepwise)search).setNumToSelect(parameter.getNumberOfFeature());
			break;
		
		case FORWARD_SELECTION_WRAPPER:
			//wrraper method with the classifier
			WrapperSubsetEval wrraperEvaluator = new WrapperSubsetEval();
			wrraperEvaluator.setClassifier(parameter.getClassifier());
			wrraperEvaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_AUC,WrapperSubsetEval.TAGS_EVALUATION));
			wrraperEvaluator.setFolds(10);
			evaluator = wrraperEvaluator;
			
			//greedy search algorithm
			search = new GreedyStepwise();
			((GreedyStepwise)search).setGenerateRanking(true);
			((GreedyStepwise)search).setNumToSelect(parameter.getNumberOfFeature());
			break;
			
		case BACKWARD_SELECTION_WRAPPER:
			//wrraper method with the classifier
			WrapperSubsetEval BCKwrraperEvaluator = new WrapperSubsetEval();
			BCKwrraperEvaluator.setClassifier(parameter.getClassifier());
			BCKwrraperEvaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_AUC,WrapperSubsetEval.TAGS_EVALUATION));
			BCKwrraperEvaluator.setFolds(10);
			evaluator = BCKwrraperEvaluator;
			
			//greedy search algorithm
			search = new GreedyStepwise();
			((GreedyStepwise)search).setSearchBackwards(true);
			((GreedyStepwise)search).setGenerateRanking(true);
			((GreedyStepwise)search).setNumToSelect(parameter.getNumberOfFeature());
			break;
		}
		
		//wrap the evaluator and the search algorithm
		attributeSelection.setEvaluator(evaluator);
		attributeSelection.setSearch(search);
		attributeSelection.setInputFormat(parameter.getFormatData());

		//return the filter
		return attributeSelection;
	}
}


