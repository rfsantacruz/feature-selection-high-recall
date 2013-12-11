package evaluation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FoldResult {

	private Map<EClassificationMetric, Double> metricsReported;
	private String optimalSetting;
	private int[] selectedFeatures;
	
	public FoldResult( ) {
		this.metricsReported = new HashMap<EClassificationMetric, Double>();
	}
	public FoldResult(String optimalString ) {
		this.metricsReported = new HashMap<EClassificationMetric, Double>();
		this.optimalSetting = optimalString;
	}
	
	public void setMetricReported(EClassificationMetric metric, double metricValue){
		this.metricsReported.put(metric, metricValue);
	}
	public double getMetricReported(EClassificationMetric metric){
		Double value = this.metricsReported.get(metric); 
		return value != null? value.doubleValue() : 0;
	}
	
	public void addSelectedFeature(int[]  idxs){
		this.selectedFeatures = idxs;
	}
	public int[] getFeatureSelected(){
		return this.selectedFeatures;
	}

	public String getOptimalSetting() {
		return optimalSetting;
	}

	public void setOptimalSetting(String optimalSetting) {
		this.optimalSetting = optimalSetting;
	}
	
}
