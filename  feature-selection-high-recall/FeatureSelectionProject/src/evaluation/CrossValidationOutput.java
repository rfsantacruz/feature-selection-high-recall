package evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;


import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import com.google.common.base.Function;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Primitives;



public class CrossValidationOutput {

	private List<FoldResult> foldsResults;
	private long seed;
	private int folds;
	private static final DecimalFormat  formatter = new DecimalFormat ("#0.000");
	
	//functions to simplify code
	private static Function<FoldResult, Double> getAccs = new Function<FoldResult, Double>(){
		public Double apply(FoldResult foldResut) {
		    return foldResut.getAccuracy();
		  }
	};
	private static Function<FoldResult, Double> getPrecs = new Function<FoldResult, Double>(){
		public Double apply(FoldResult foldResut) {
		    return foldResut.getPrecision();
		  }
	};
	private static Function<FoldResult, Double> getRecs = new Function<FoldResult, Double>(){
		public Double apply(FoldResult foldResut) {
		    return foldResut.getRecall();
		  }
	};
	private static Function<FoldResult, Double> getfms = new Function<FoldResult, Double>(){
		public Double apply(FoldResult foldResut) {
		    return foldResut.getFmeasure();
		  }
	};
	

	public CrossValidationOutput(long seed,
			int folds) {

		super();
		this.foldsResults = new ArrayList<FoldResult>();
		this.seed = seed;
		this.folds = folds;
	}

	public double accuracyStd(){
		
		double[] accuracys = Doubles.toArray(Lists.transform(this.foldsResults, getAccs));
		return Math.sqrt(StatUtils.variance(accuracys));
	}
	public double precisionStd(){
		
		double[] precisions = Doubles.toArray(Lists.transform(this.foldsResults, getPrecs));
		return Math.sqrt(StatUtils.variance(precisions));
	}
	public double recallyStd(){
		double[] recalls = Doubles.toArray(Lists.transform(this.foldsResults, getRecs));
		return Math.sqrt(StatUtils.variance(recalls));
	}
	public double fmeasureStd(){
		double[] fms = Doubles.toArray(Lists.transform(this.foldsResults, getfms));
		return Math.sqrt(StatUtils.variance(fms));
	}
	public double accuracyMean(){
		double[] accs = Doubles.toArray(Lists.transform(this.foldsResults, getAccs));
		return StatUtils.mean(accs);
	}
	public double precisionMean(){
		double[] precs = Doubles.toArray(Lists.transform(this.foldsResults, getPrecs));
		return StatUtils.mean(precs);
	}
	public double recallMean(){
		double[] recs = Doubles.toArray(Lists.transform(this.foldsResults, getRecs));
		return StatUtils.mean(recs);
	}
	public double fmeasureMean(){
		double[] fms = Doubles.toArray(Lists.transform(this.foldsResults, getfms));
		return StatUtils.mean(fms);
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
