package evaluation;

public class FoldResult {

	private double accuracy;
	private double precision;
	private double recall;
	private double fmeasure;
	private String optimalSetting; 
	
	public FoldResult(double accuracy, double precision, double recall,
			double fmeasure, String optimalSetting) {
		super();
		this.setAccuracy(accuracy);
		this.precision = precision;
		this.recall = recall;
		this.fmeasure = fmeasure;
		this.setOptimalSetting(optimalSetting);
	}
	
	public FoldResult( ) {}

	public double getAccuracy() {
		return accuracy;
	}

	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
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

	public double getFmeasure() {
		return fmeasure;
	}

	public void setFmeasure(double fmeasure) {
		this.fmeasure = fmeasure;
	}

	public String getOptimalSetting() {
		return optimalSetting;
	}

	public void setOptimalSetting(String optimalSetting) {
		this.optimalSetting = optimalSetting;
	}
	
}
