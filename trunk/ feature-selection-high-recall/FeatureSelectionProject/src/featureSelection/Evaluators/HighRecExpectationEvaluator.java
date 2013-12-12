package featureSelection.Evaluators;

import java.util.BitSet;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class HighRecExpectationEvaluator extends OurBaseFeatureSelectionEvaluator{

	//analyse subsets of attributes
	@Override
	public double evaluateSubset(BitSet subSet) throws Exception {

		//avoid evalaluate the empty set 
		if(subSet.cardinality() == 0)
			return 0;

		//score
		double score = 0;
		int numOfClasses = super.dataBinarized.numClasses();

		//for each datum of the data set
		for (Instance datum : super.dataBinarized) {

			//for all classes
			for(int yclass = 0; yclass < numOfClasses; yclass++){
				//main probability of the equations
				double p = 1;
				//productory of P(y_i=1|x_i, f_i) for this datum. obs these probs were precomputed
				//run features in the subset
				for (int i = subSet.nextSetBit(0); i >= 0; i = subSet.nextSetBit(i + 1)) {
					int xdi = (int)datum.toDoubleArray()[i];
					//P(y_i != c|x_i, f_i) = 1 - P(y_i = c|x_i, f_i)
					p = p * (1 - this.probs[i][yclass][xdi]);
				}

				//indexers in the equation
				int indexerPos = datum.classValue() == yclass ? 1 : 0;
				int indexerNeg = datum.classValue() != yclass ? 1 : 0;

				//sum
				score += (indexerPos * (1-p)) + (indexerNeg *  p);
			}
		}

		return score;
	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}

}
