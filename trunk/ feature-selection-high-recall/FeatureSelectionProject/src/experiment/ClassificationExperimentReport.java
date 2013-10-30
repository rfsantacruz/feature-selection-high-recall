package experiment;

import java.text.DecimalFormat;

//class to save the metrics generated
public class ClassificationExperimentReport extends AbstractExperimentReport{

	private double precision;
	private double recall;
	private double accuracy;
	private double f_measure;
	private String problemName;
	private String classifierName;
	private static final DecimalFormat  formatter = new DecimalFormat ("#0.000"); 

	
	
	public ClassificationExperimentReport(double precision, double recall, double accuracy,
			double f_measure, String problemName, String classifierName) {
		this.precision = precision;
		this.recall = recall;
		this.accuracy = accuracy;
		this.f_measure = f_measure;
		this.setProblemName(problemName);
		this.setClassifierName(classifierName);
	}

	public ClassificationExperimentReport(String problemName, String classifierName) {
		this.problemName = problemName;
		this.classifierName = classifierName;
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
	public String getProblemName() {
		return problemName;
	}
	public void setProblemName(String problemName) {
		this.problemName = problemName;
	}
	public String getClassifierName() {
		return classifierName;
	}
	public void setClassifierName(String classifierName) {
		this.classifierName = classifierName;
	}

	@Override
	public String toString() {
		return "ExperimentReport [precision=" + formatter.format(precision) + ", recall="
				+ formatter.format(recall) + ", accuracy=" + formatter.format(accuracy) + ", f_measure="
				+ formatter.format(f_measure) + ", problemName=" + problemName
				+ ", classifierName=" + classifierName + "]";
	}

	private String toCSV(){
		return 	toCSV(";");
	}
	private String toCSV(String sep){
		return problemName + sep + classifierName + sep + formatter.format(accuracy) + sep + formatter.format(precision) + sep + formatter.format(recall) + sep
				+ formatter.format(f_measure);		
	}

	@Override
	public String outPutRepresentation() {
		// TODO: implement to plot matlab graphs
		return this.toCSV();
	}


}