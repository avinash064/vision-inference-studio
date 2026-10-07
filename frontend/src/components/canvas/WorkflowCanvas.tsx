import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, useReactFlow, type IsValidConnection } from '@xyflow/react'
import { useCallback, useState, type DragEvent, type MouseEvent } from 'react'
import { useStudio } from '../../store/studio'
import type { StudioEdge } from '../../types'
import { StudioNode } from '../nodes/StudioNode'

const nodeTypes = { studio: StudioNode }

export function WorkflowCanvas() {
  const { screenToFlowPosition, fitView } = useReactFlow()
  const { nodes, edges, definitions, models, onNodesChange, onEdgesChange, connect, addNode, select } = useStudio()
  const [picker, setPicker] = useState<{ x: number; y: number }>()
  const drop = useCallback((event: DragEvent) => { event.preventDefault(); const type = event.dataTransfer.getData('application/vision-node'); if (type) addNode(type, screenToFlowPosition({ x: event.clientX, y: event.clientY })) }, [addNode, screenToFlowPosition])
  const doubleClick = (event: MouseEvent) => setPicker({ x: event.clientX, y: event.clientY })
  const validConnection: IsValidConnection<StudioEdge> = (connection) => {
    const source = nodes.find((node) => node.id === connection.source), target = nodes.find((node) => node.id === connection.target)
    const sourceDef = definitions.find((item) => item.type === source?.data.nodeType), targetDef = definitions.find((item) => item.type === target?.data.nodeType)
    const output = sourceDef?.outputs.find((port) => port.id === connection.sourceHandle), input = targetDef?.inputs.find((port) => port.id === connection.targetHandle)
    let types = output?.data_types || []
    if (source?.data.nodeType === 'model') { const upstream = edges.find((edge) => edge.target === source.id); const upstreamNode = nodes.find((node) => node.id === upstream?.source); types = upstreamNode?.data.nodeType === 'video_source' ? ['video'] : [models?.families[String(source.data.config.family || '')]?.result_type || 'tensor'] }
    return !!output && !!input && types.some((type) => input.data_types.includes(type))
  }
  return <main className="canvas-wrap" onDoubleClick={doubleClick}>
    <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={connect} isValidConnection={validConnection} onDrop={drop} onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }} onNodeClick={(_, node) => select(node.id)} onPaneClick={() => { select(); setPicker(undefined) }} deleteKeyCode={['Backspace', 'Delete']} multiSelectionKeyCode="Shift" fitView>
      <Background variant={BackgroundVariant.Dots} gap={24} size={1.4} color="#344056" />
      <Controls position="bottom-left" onFitView={() => fitView()} />
      <MiniMap position="bottom-right" pannable zoomable nodeColor="#7857ff" maskColor="rgba(7,10,17,.76)" />
    </ReactFlow>
    {nodes.length === 0 && <div className="empty-canvas"><span>01</span><h2>Compose your first inference graph</h2><p>Drag an input, connect a model, and finish with a preview.</p></div>}
    {picker && <div className="quick-picker" style={{ left: picker.x, top: picker.y }} onDoubleClick={(event) => event.stopPropagation()}>{definitions.map((item) => <button key={item.type} onClick={() => { addNode(item.type, screenToFlowPosition(picker)); setPicker(undefined) }}>{item.label}<small>{item.category}</small></button>)}</div>}
  </main>
}
