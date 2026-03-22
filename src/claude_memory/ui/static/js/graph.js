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
        this.$nextTick(() => this.render());
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

      const width = container.clientWidth;
      const height = container.clientHeight || 500;

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
      const palette = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#64748b'];
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
        .attr('stroke', '#334155')
        .attr('stroke-opacity', 0.4)
        .attr('stroke-width', 1);

      const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => 5 + d.importance * 15)
        .attr('fill', d => categoryColors[d.category] || '#64748b')
        .attr('stroke', '#1e293b')
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

      simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
        node
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);
      });
    },

    preview(content, len = 200) {
      if (!content) return '';
      return content.length > len ? content.substring(0, len) + '...' : content;
    },
  };
}
