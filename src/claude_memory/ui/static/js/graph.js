// D3.js force-directed graph visualization
function graphComponent() {
  return {
    memories: [],
    categories: [],
    selectedCategories: {},
    loading: false,
    selectedNode: null,
    simulation: null,

    async init() {
      this.loading = true;
      try {
        const [memData, catData] = await Promise.all([
          api.get('/api/memories?limit=200'),
          api.get('/api/categories'),
        ]);
        this.memories = memData.memories;
        this.categories = catData.categories;
        this.categories.forEach(c => this.selectedCategories[c] = true);
        setTimeout(() => this.render(), 100);
      } catch (e) {
        console.error('Failed to load graph data:', e);
      }
      this.loading = false;
    },

    toggleCategory(cat) {
      this.selectedCategories[cat] = !this.selectedCategories[cat];
      this.render();
    },

    getFilteredMemories() {
      return this.memories.filter(m => this.selectedCategories[m.category]);
    },

    render() {
      const container = this.$refs.graphContainer;
      if (!container) return;

      // Clear previous
      d3.select(container).selectAll('*').remove();
      if (this.simulation) this.simulation.stop();

      const memories = this.getFilteredMemories().slice(0, 300);
      if (memories.length === 0) return;

      const width = container.clientWidth || container.parentElement?.clientWidth || 800;
      const height = Math.max(container.clientHeight, window.innerHeight - 180);

      // Build nodes and tag-based edges
      const nodes = memories.map(m => ({
        id: m.id,
        content: m.content,
        category: m.category,
        tags: m.tags,
        importance: m.importance,
        is_sensitive: m.is_sensitive,
      }));

      const tagMap = {};
      nodes.forEach(n => {
        if (!n.tags) return;
        n.tags.split(',').map(t => t.trim()).filter(Boolean).forEach(tag => {
          if (!tagMap[tag]) tagMap[tag] = [];
          tagMap[tag].push(n.id);
        });
      });

      const links = [];
      const linkSet = new Set();
      Object.values(tagMap).forEach(ids => {
        for (let i = 0; i < ids.length && i < 5; i++) {
          for (let j = i + 1; j < ids.length && j < 5; j++) {
            const key = `${ids[i]}-${ids[j]}`;
            if (!linkSet.has(key)) {
              linkSet.add(key);
              links.push({ source: ids[i], target: ids[j] });
            }
          }
        }
      });

      const categoryColors = {};
      const palette = ['#d4a04a', '#7a8b6f', '#c4785a', '#8b7355', '#6b8e8e', '#b8860b', '#9aad82', '#d4764a', '#7a6b55', '#5f8a8a'];
      this.categories.forEach((c, i) => categoryColors[c] = palette[i % palette.length]);

      const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

      const g = svg.append('g');

      // Zoom
      const zoom = d3.zoom()
        .scaleExtent([0.2, 5])
        .on('zoom', (event) => g.attr('transform', event.transform));
      svg.call(zoom);

      const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide().radius(d => 8 + d.importance * 15));

      this.simulation = simulation;

      const link = g.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', '#3a3228')
        .attr('stroke-opacity', 0.5)
        .attr('stroke-width', 1);

      const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => 5 + d.importance * 15)
        .attr('fill', d => categoryColors[d.category] || '#8b7355')
        .attr('stroke', '#252019')
        .attr('stroke-width', 1.5)
        .attr('cursor', 'pointer')
        .on('click', (event, d) => {
          this.selectedNode = d;
        })
        .call(d3.drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x; d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          }));

      node.append('title').text(d => d.content ? d.content.substring(0, 60) : '');

      // Node labels for high-importance memories
      const labels = g.append('g')
        .selectAll('text')
        .data(nodes.filter(d => d.importance >= 0.7))
        .join('text')
        .attr('font-size', '10px')
        .attr('fill', '#c4b8a8')
        .attr('font-family', "'JetBrains Mono', monospace")
        .attr('pointer-events', 'none')
        .text(d => d.content ? d.content.substring(0, 25) + '\u2026' : '');

      // Legend (appended to svg, not g, so it stays fixed during zoom/pan)
      const legend = svg.append('g')
        .attr('class', 'graph-legend')
        .attr('transform', 'translate(16, 16)');

      const cats = Object.entries(categoryColors);
      cats.forEach(([cat, color], i) => {
        const row = legend.append('g').attr('transform', `translate(0, ${i * 22})`);
        row.append('circle').attr('r', 6).attr('cx', 6).attr('cy', 6).attr('fill', color);
        row.append('text').attr('x', 18).attr('y', 10)
          .attr('fill', '#c4b8a8')
          .attr('font-size', '12px')
          .attr('font-family', "'JetBrains Mono', monospace")
          .text(cat);
      });

      // Semi-transparent background behind legend
      const legendBBox = legend.node().getBBox();
      legend.insert('rect', ':first-child')
        .attr('x', legendBBox.x - 8)
        .attr('y', legendBBox.y - 8)
        .attr('width', legendBBox.width + 16)
        .attr('height', legendBBox.height + 16)
        .attr('rx', 6)
        .attr('fill', '#1a1613')
        .attr('fill-opacity', 0.85);

      simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
        node
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);
        labels
          .attr('x', d => d.x + 8 + d.importance * 15)
          .attr('y', d => d.y + 4);
      });
    },

    preview(content, len = 200) {
      if (!content) return '';
      return content.length > len ? content.substring(0, len) + '...' : content;
    },
  };
}
