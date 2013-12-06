package featureSelection;

import java.util.BitSet;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class HighPreLogLikelihoodEvaluator extends ASEvaluation implements SubsetEvaluator  {

	
	private Instances dataBinarized;
	//probs[i][yd][xi] = P(y_i  | xi, fi)
	private double[][][] probs;
	//laplace smoth paramter
	private double lspValue = 1;

	
	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {

		//bunarize the data set
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		disTransform.setMakeBinary(true);
		this.dataBinarized = Filter.useFilter(data, disTransform);
		
		//parameter to do laplace smoothing of the conditional probability
		int multplier = this.dataBinarized.attribute(this.dataBinarized.classIndex()).numValues();
		
		//compute conditional probabilities
		//probs[i][yd][xi] = P(y_i = 1 | xi, fi)
		this.probs = new double[this.dataBinarized.numAttributes() - 1][2][2];

		for(int f = 0; f < this.probs.length; f++){

			//Gets the value of all instances for the attribute f  
			double[]  fAttibuteValues = this.dataBinarized.attributeToDoubleArray(f);
			double[] ydLabels = this.dataBinarized.attributeToDoubleArray(this.dataBinarized.classIndex());
			
			//frequency counts
			double count_xi0 = 0;
			double count_xi1 = 0;
			double count_y1_xi0 = 0;
			double count_y1_xi1 = 0;
			double count_y0_xi0 = 0;
			double count_y0_xi1 = 0;
			
			for (int d = 0; d < fAttibuteValues.length; d++ ) {
				double xdi = fAttibuteValues[d];
				double yd = ydLabels[d];
				
				//count  xi = 0
				if(xdi == 0){
					count_xi0++;
					//count  xi = 0 and yd = 1
					if(yd == 1){
						count_y1_xi0++;
					}else{
						//count xi = 0 and yd = 0
						count_y0_xi0++;
					}
				}else{
					//count xi = 1
					count_xi1++;
					//count xi=1 and yd =1
					if(yd == 1){
						count_y1_xi1++;
					}else{
						//count xi = 1 and yd = 0
						count_y0_xi1++;
					}
				}

			}
			
			
			
			//compute the probability and peform laplace smoothing to avoid small values
			this.probs[f][0][0] = (lspValue + count_y0_xi0)/(count_xi0 + lspValue * multplier );
			this.probs[f][0][1]= (lspValue + count_y0_xi1)/(count_xi1 + lspValue * multplier);
			this.probs[f][1][0] = (lspValue + count_y1_xi0)/(count_xi0 + lspValue * multplier);
			this.probs[f][1][1]= (lspValue + count_y1_xi1)/(count_xi1 + lspValue * multplier);
			
			/*//debug - this properties have to be true unless you create another math
			System.out.println("is equal 1 ?" + (this.probs[f][0][0] + this.probs[f][1][0]));
			System.out.println("is equal 1 ?" + (this.probs[f][0][1] + this.probs[f][1][1]));
			System.out.println("is equal " +this.dataBinarized.size() +" ?" + (count_y0_xi0 + count_y0_xi1 + count_y1_xi0 + count_y1_xi1));
			*/
		}

	}

	//analyse subsets of attributes
	@Override
	public double evaluateSubset(BitSet subSet) throws Exception {

		//avoid evalaluate the empty set 
		if(subSet.cardinality() == 0)
			return 0;

		//score
		double score = 0;

		//for each datum of the data set
		for (Instance datum : this.dataBinarized) {
			//main probability of the equations
			double p = 1;
			//productory of P(y_i=1|x_i, f_i) for this datum. obs these probs were precomputed
			for (int i = 0; i < this.dataBinarized.numAttributes(); i++) {
				if(subSet.get(i)){
					int xdi = (int)datum.toDoubleArray()[i];
					//P(y_i = 1|x_i, f_i)
					p = p * this.probs[i][1][xdi];
				}
			}
			//indexers in the equation
			int indexerPos = datum.classValue() == 1 ? 1 : 0;
			int indexerNeg = datum.classValue() == 0 ? 1 : 0;

			//sum
			score += (indexerPos * Math.log(p)) + (indexerNeg * Math.log((1 - p)));
		}

		return score;
	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}
}
