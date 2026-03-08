import React from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

export default function GraphView({ graphData, selectedNode, onNodeClick }) {

    // Map graph data to Cytoscape format
    const elements = [];

    if (graphData && graphData.nodes) {
        graphData.nodes.forEach(node => {
            // Determine color based on calculated risk
            let bgColor = '#10B981'; // Green (LOW)
            if (node.calculated_risk_score >= 0.5) bgColor = '#EF4444'; // Red (CRITICAL)
            else if (node.calculated_risk_score >= 0.3) bgColor = '#F97316'; // Orange (HIGH)
            else if (node.calculated_risk_score > 0.0) bgColor = '#EAB308'; // Yellow (MODERATE)

            elements.push({
                data: { id: node.id, label: node.name, risk: node.calculated_risk_score, bgColor: bgColor }
            });
        });

        graphData.edges.forEach(edge => {
            elements.push({
                data: {
                    source: edge.source,
                    target: edge.target,
                    amp: edge.amplification_factor
                }
            });
        });
    }

    const layout = {
        name: 'circle',
        padding: 50,
        animate: true,
        animationDuration: 500,
    };

    const stylesheet = [
        {
            selector: 'node',
            style: {
                'label': 'data(label)',
                'color': '#fff',
                'background-color': 'data(bgColor)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '12px',
                'width': '50px',
                'height': '50px',
                'border-width': 2,
                'border-color': '#1f2937'
            }
        },
        {
            selector: 'edge',
            style: {
                'width': 2,
                'line-color': '#4b5563',
                'target-arrow-color': '#4b5563',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 1.5
            }
        },
        {
            selector: ':selected',
            style: {
                'border-width': 4,
                'border-color': '#3b82f6',
            }
        }
    ];

    return (
        <div className="w-full h-full bg-gray-900 rounded-lg border border-gray-800 overflow-hidden relative">
            <CytoscapeComponent
                elements={elements}
                style={{ width: '100%', height: '100%' }}
                stylesheet={stylesheet}
                layout={layout}
                cy={(cy) => {
                    cy.on('tap', 'node', function (evt) {
                        const node = evt.target;
                        if (onNodeClick) onNodeClick(node.data('id'));
                    });
                    // Force layout recalculation when elements change
                    cy.layout(layout).run();
                    cy.fit();
                }}
            />
            {elements.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    No graph data initialized.
                </div>
            )}
        </div>
    );
}
