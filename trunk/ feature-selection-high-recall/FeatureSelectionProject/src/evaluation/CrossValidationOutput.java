package evaluation;

import java.text.DecimalFormat;

public class CrossValidationOutput {
	
	private double precision;
	private double recall;
	private double accuracy;
	private double f_measure;
	
	private static final DecimalFormat  formatter = new DecimalFormat ("#0.000"); 

	
	public CrossValidationOutput(double precision, double recall, double accuracy,
			double f_measure) {
		
		this.precision = precision;
		this.recall = recall;
		this.accuracy = accuracy;
		this.f_measure = f_measure;

	}


	public double getPrecision() {
		return precision;
	}
	public void setPrecision(double precision) {
		this.precision = precision;
	}
	public double getRecall() {
		return recall;
	}
	public void setRecall(double recall) {
		this.recall = recall;
	}
	public double getAccuracy() {
		return accuracy;
	}
	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}
	public double getF_measure() {
		return f_measure;
	}
	public void setF_measure(double f_measure) {
		this.f_measure = f_measure;
	}
	

	@Override
	public String toString() {
		return "CrossValidationOutput [precision=" + formatter.format(precision) + ", recall="
				+ formatter.format(recall) + ", accuracy=" + formatter.format(accuracy) + ", f_measure="
				+ formatter.format(f_measure) + "]";
	}


	
}
