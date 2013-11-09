package experiment;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.text.StrBuilder;


import com.google.common.base.Joiner;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

public class FeatureSelectionExperimentReport extends AbstractExperimentReport {

	private Map<String, List<Double>> featureSelection2metric;
	private String classifier;
	private String problem;
	private int maxNumFeatures;
	private String metricName;

	@Override
	public void saveInFile(String path) {
		String fileName = Joiner.on("_").skipNulls().join(this.problem, this.classifier, this.metricName, ".m" );
		Path file = Paths.get(path,fileName);

		try(PrintWriter out = new PrintWriter(file.toFile())){

			out.print(this.saveString());			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	@Override
	public String saveString() {
		
		StrBuilder sb = new StrBuilder();
		sb.appendln(Joiner.on(",").skipNulls().join(this.problem, this.classifier, this.metricName,this.maxNumFeatures));

		for (String alg : featureSelection2metric.keySet()) {
			if(featureSelection2metric.get(alg) != null){
				sb.append(Joiner.on(",").skipNulls().join(alg,","));
				sb.appendln(Joiner.on(",").skipNulls().join(featureSelection2metric.get(alg)));
			}
		}
		return sb.toString();
	}
		
	@Override
	public void plot(String path) {

		String fileName = Joiner.on("_").skipNulls().join(this.problem, this.classifier, this.metricName, ".m" );
		Path file = Paths.get(path,fileName);

		try(PrintWriter out = new PrintWriter(file.toFile())){

			out.print(this.plotString());			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	@Override
	public String plotString() {

		StrBuilder sb = new StrBuilder();

		Iterator<String> plotSymbols =   Iterables.cycle(
				"'-yo'","'-mo'","'-co'","'-ro'","'-go'","'-bo'","'-wo'","'-ko'",
				"'-y+'","'-m+'","'-c+'","'-r+'","'-g+'","'-b+'","'-w+'","'-k+'",
				"'-y*'","'-m*'","'-c*'","'-r*'","'-g*'","'-b*'","'-w*'","'-k*'",
				"'-yx'","'-mx'","'-cx'","'-rx'","'-gx'","'-bx'","'-wx'","'-kx'",
				"'-ys'","'-ms'","'-cs'","'-rs'","'-gs'","'-bs'","'-ws'","'-ks'",
				"'-yd'","'-md'","'-cd'","'-rd'","'-gd'","'-bd'","'-wd'","'-kd'").iterator();



		sb.appendln("%simulation Plot");

		sb.appendln("%open plot");
		sb.appendln("figure, hold on;");
		sb.appendln("");

		sb.appendln("%Simulation results");
		sb.appendln("n_features = [1: "+ this.maxNumFeatures + "];");
		sb.appendln("");

		List<String> legend = new ArrayList<String>();
		for (String alg : this.featureSelection2metric.keySet()) {
			if(featureSelection2metric.get(alg) != null){
				String metric = "metric_" + alg;
				String error = "err_" + alg;
				legend.add("'" + alg.replaceAll("_", " ") + "'");
				String symbol = plotSymbols.next();

				sb.appendln(metric + " = ["+ Joiner.on(", ").skipNulls().join(featureSelection2metric.get(alg)) + "];");
				sb.appendln(error + " = std(" + metric + ")*ones(size(n_features));");
				sb.appendln("errorbar(n_features, "+ metric + "," + error + ","+ symbol +",'LineWidth', 1.5,'MarkerSize',8)");
				sb.appendln("");
			}
		}

		sb.appendln("%plots settings");
		sb.appendln("title('"+ Joiner.on(" ").skipNulls().join(this.problem, this.classifier, this.metricName ) +"');");
		sb.appendln("xlabel('number of features');");
		sb.appendln("ylabel('"+ this.metricName +"');");
		sb.appendln("legend(" + Joiner.on(",").skipNulls().join(legend) + ");");

		return sb.toString();
	}


	public FeatureSelectionExperimentReport() {
		super();
	}

	public FeatureSelectionExperimentReport(
			Map<String, List<Double>> featureSelection2metric,
			String classifier, String problem, int maxNumFeatures,
			String metricName) {
		super();
		this.featureSelection2metric = featureSelection2metric;
		this.classifier = classifier;
		this.problem = problem;
		this.maxNumFeatures = maxNumFeatures;
		this.metricName = metricName;
	}

	public Map<String, List<Double>> getFeatureSelection2metric() {
		return featureSelection2metric;
	}

	public void setFeatureSelection2metric(Map<String, List<Double>> featureSelection2metric) {
		this.featureSelection2metric = featureSelection2metric;
	}

	public String getClassifier() {
		return classifier;
	}

	public void setClassifier(String classifier) {
		this.classifier = classifier;
	}

	public String getProblem() {
		return problem;
	}

	public void setProblem(String problem) {
		this.problem = problem;
	}

	public int getMaxNumFeatures() {
		return maxNumFeatures;
	}

	public void setMaxNumFeatures(int maxNumFeatures) {
		this.maxNumFeatures = maxNumFeatures;
	}

	public String getMetricName() {
		return metricName;
	}

	public void setMetricName(String metricName) {
		this.metricName = metricName;
	}



}
