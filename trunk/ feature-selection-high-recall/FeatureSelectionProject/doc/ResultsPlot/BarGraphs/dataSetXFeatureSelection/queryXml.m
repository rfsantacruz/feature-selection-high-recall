function [ nodeList ] = queryXml( xpathExp, file )

% Import the XPath classes
import javax.xml.xpath.*

% Construct the DOM.
doc = xmlread(which(file));

% Create an XPath expression.
factory = XPathFactory.newInstance;
xpath = factory.newXPath;
expression = xpath.compile(xpathExp);

% Apply the expression to the DOM.
nodeList = expression.evaluate(doc,XPathConstants.NODESET);

% Iterate through the nodes that are returned.
for i = 1:nodeList.getLength
    node = nodeList.item(i-1);
    disp(char(node.getAttribute('name')))
end;

end

