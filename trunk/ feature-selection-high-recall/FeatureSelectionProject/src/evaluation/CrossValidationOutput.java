package evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;


import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;



public class CrossValidationOutput {

	private List<FoldResult> foldsResults;
	private long seed;
	private int folds;
	private static final DecimalFormat  formatter = new DecimalFormat ("#0.000");



	public CrossValidationOutput(long seed,
			int folds) {

		super();
		this.foldsResults = new ArrayList<FoldResult>();
		this.seed = seed;
		this.folds = folds;
	}

	
	
	public double accuracyStd(){
		return Math.sqrt(StatUtils.variance(this.accuraccyVector()));
	}
	public double precisionStd(){
		return Math.sqrt(StatUtils.variance(this.precisionVector()));
	}
	public double recallyStd(){
		return Math.sqrt(StatUtils.variance(this.recallVector()));
	}
	public double fmeasureStd(){
		return Math.sqrt(StatUtils.variance(this.fmeasureVector()));
	}
	public double accuracyMean(){
		return StatUtils.mean(this.accuraccyVector());
	}
	public double precisionMean(){
		return StatUtils.mean(this.precisionVector());
	}
	public double recallMean(){
		return StatUtils.mean(this.recallVector());
	}
	public double fmeasureMean(){
		return StatUtils.mean(this.fmeasureVector());
	}
	
	
	//private
	private double[] accuraccyVector(){
		double[] ret = new double[this.foldsResults.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = this.foldsResults.get(i).getAccuracy();
		}
		return ret;
	}
	private double[] precisionVector(){
		double[] ret = new double[this.foldsResults.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = this.foldsResults.get(i).getPrecision();
		}
		return ret;
	}
	private double[] recallVector(){
		double[] ret = new double[this.foldsResults.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = this.foldsResults.get(i).getRecall();
		}
		return ret;
	}
	private double[] fmeasureVector(){
		double[] ret = new double[this.foldsResults.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = this.foldsResults.get(i).getFmeasure();
		}
		return ret;
	}
	
	

	//getters and setters
	public List<FoldResult> getFoldsResults() {
		return foldsResults;
	}
	public FoldResult getFoldResult(int index) {
		return foldsResults.get(index);
	}

	public void addFoldResult(FoldResult foldsResult) {
		this.foldsResults.add(foldsResult);
	}

	public long getSeed() {
		return seed;
	}

	public void setSeed(long seed) {
		this.seed = seed;
	}

	public int getFolds() {
		return folds;
	}

	public void setFolds(int folds) {
		this.folds = folds;
	}


	@Override
	public String toString() {
		return "CrossValidationOutput [accuracyMean()=" + formatter.format(accuracyMean())
				+ ", precisionMean()=" + formatter.format(precisionMean()) + ", recallMean()="
				+ formatter.format(recallMean()) + ", fmeasureMean()=" + formatter.format(fmeasureMean())
				+ ", accuracyStd()=" + formatter.format(accuracyStd()) + ", precisionStd()="
				+ formatter.format(precisionStd()) + ", recallyStd()=" + formatter.format(recallyStd())
				+ ", fmeasureStd()=" + formatter.format(fmeasureMean()) + ", seed=" + seed
				+ ", folds=" + folds + "]";
	}






}
