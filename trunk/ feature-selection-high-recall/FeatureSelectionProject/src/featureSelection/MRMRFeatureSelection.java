package featureSelection;

import java.util.BitSet;

import JavaMI.MutualInformation;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class MRMRFeatureSelection extends ASEvaluation implements SubsetEvaluator   {
	
	private Instances dataDiscretized;
	private double[] relevance;
	private double[][] redundancy;
	
	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {
		
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataDiscretized = Filter.useFilter(data, disTransform);
		
		//relevance precomputing mutual information feature-label
		this.relevance = new double[this.dataDiscretized.numAttributes()];
		double[] classATT = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		for (int i = 0; i < this.relevance.length; i++) {
			double[] featureI = this.dataDiscretized.attributeToDoubleArray(i);
			relevance[i] = MutualInformation.calculateMutualInformation(classATT, featureI);
		}
		
		//redundancy precomputing mutual information feature-feature
		this.redundancy = new double[this.dataDiscretized.numAttributes()][this.dataDiscretized.numAttributes()];
		for (int i = 0; i < redundancy.length; i++) {
			double[] featureI = this.dataDiscretized.attributeToDoubleArray(i);
			for (int j = 0; j < redundancy[i].length; j++) {
				double[] featureJ = this.dataDiscretized.attributeToDoubleArray(j);
				redundancy[i][j] = MutualInformation.calculateMutualInformation(featureI,featureJ);
			}
		}
		
		

	}

	//analyse subsets of attributes
	@Override
	public double evaluateSubset(BitSet subSet) throws Exception {

		if(subSet.cardinality() == 0)
			return 0;
		
		double relevance = 0.0;
		double redundancy = 0.0;
		double[] classATT = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		
		for (int i = 0; i < this.dataDiscretized.numAttributes(); i++) {
			if(subSet.get(i)){
				relevance += this.relevance[i];
				for (int j = 0; j < this.dataDiscretized.numAttributes(); j++) {
					if(i != j && subSet.get(j)){
						redundancy += this.redundancy[i][j];
					}
				}
				
			}
		}
		
		relevance = relevance / subSet.cardinality(); 
		redundancy = redundancy / Math.pow(subSet.cardinality(), 2);
		
		return relevance - redundancy;
	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}

}
