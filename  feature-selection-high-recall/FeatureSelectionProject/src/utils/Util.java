package utils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import org.apache.commons.io.FileUtils;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import problems.ClassificationProblem;
import experiment.ExperimentReport;


public class Util {
	

	public static List<ClassificationProblem> readAllFilesARFFFromDirectory(String path){
		List<ClassificationProblem> problems = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				String ext = com.google.common.io.Files.getFileExtension(file.toString());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION))
					problems.add(new ClassificationProblem(file.toFile().getPath()));
			}
		} catch (IOException | DirectoryIteratorException x) {
			x.printStackTrace();
		}

		System.out.println(problems.size() +" ARFF files were read");
		return problems;
	}

	public static List<ClassificationProblem> readAllFilesARFFFromJar(String path){
		List<ClassificationProblem> problems = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try {
			JarFile file = new JarFile(dir.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry)) ;
					problems.add(new ClassificationProblem(entry.getName(),inpr));
				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println(problems.size() +" ARFF files were read");
		return problems;
	}

	public static void saveExperimentReportAsCSV(String filePath, List<ExperimentReport> reports, String sep){
		Path file = Paths.get(filePath);
		try(PrintWriter out = new PrintWriter(file.toFile())){
			for (ExperimentReport report : reports) {
				if(sep == null)
					out.println(report.toCSV());
				else
					out.println(report.toCSV(sep));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void saveExperimentReportAsXML(String filePath, List<ExperimentReport> reports){
		//TODO
	}
	
	//generate all possibles models based on param
	public static List<String> generateModels(List<Set<String>> paramsList){
		List<String> modelsStringSetting = null;
		final Joiner joiner = Joiner.on(" ").skipNulls();
		Function<List<String>, String> buildModelSettingString = new Function<List<String>, String>() {
			public String apply(List<String> params) {
			    return joiner.join(params);
			}
		};
		
		if(paramsList != null && paramsList.size() > 1){
			Set<List<String>> cartesianProd = Sets.cartesianProduct(paramsList);
			
			if(cartesianProd != null){
				modelsStringSetting = Lists.newArrayList( Iterables.transform(cartesianProd, buildModelSettingString));
			}
		}
		
		return modelsStringSetting;
	}
	
	public static Set<String> generateModelsStringSettings(String settingCode, double... values){
		
		Set<String> ret = new HashSet<String>();
		for (double d : values) {
			ret.add(settingCode + " " + d);
		}
		return ret;
	}

}
