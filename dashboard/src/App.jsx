import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Sample data structure - in production, load from your JSON file
const sampleData = {
  results: [
    {
      temu_id: "601099817564070",
      title: "Interactive Baby Elephant - Christmas Gift Interactive Baby Elephant",
      price: 16.13,
      image_path: "temu_baby_toys_imgs/601099817564070.jpg",
      analysis: {
        viability_score: 8,
        product_quality_score: 7,
        market_demand_score: 9,
        verdict: "RESEARCH FURTHER",
        competition_level: "Medium",
        safety_compliance_risks: { risk_level: "Low" },
        estimated_amazon_price_range: { low: 34.99, high: 49.99 },
        primary_concerns: ["Competition", "Seasonality"],
        verdict_reasoning: "Strong demand with good profit margins"
      }
    },
    {
      temu_id: "601100041101610",
      title: "54 In 1 Busy Board Montessori Toys For Toddler",
      price: 15.64,
      image_path: "temu_baby_toys_imgs/601100041101610.jpg",
      analysis: {
        viability_score: 6,
        product_quality_score: 5,
        market_demand_score: 8,
        verdict: "RESEARCH FURTHER",
        competition_level: "High",
        safety_compliance_risks: { risk_level: "Medium" },
        estimated_amazon_price_range: { low: 24.99, high: 39.99 },
        primary_concerns: ["Safety Compliance", "Competition", "Quality Control"],
        verdict_reasoning: "Popular category but requires careful supplier vetting"
      }
    },
    {
      temu_id: "601100123456",
      title: "LED Light Up Drawing Board for Kids",
      price: 12.99,
      image_path: "temu_baby_toys_imgs/601100123456.jpg",
      analysis: {
        viability_score: 4,
        product_quality_score: 4,
        market_demand_score: 5,
        verdict: "SKIP",
        competition_level: "High",
        safety_compliance_risks: { risk_level: "Low" },
        estimated_amazon_price_range: { low: 19.99, high: 24.99 },
        primary_concerns: ["Low margins", "Saturated market"],
        verdict_reasoning: "Margins too thin after Amazon fees"
      }
    }
  ]
};

const ProductDashboard = () => {
  const [products, setProducts] = useState([]);
  const [filterVerdict, setFilterVerdict] = useState('all');
  const [sortBy, setSortBy] = useState('viability');
  const [minViability, setMinViability] = useState(0);
  const [selectedProducts, setSelectedProducts] = useState([]);
  const [viewMode, setViewMode] = useState('grid');

  // Load data (in production, fetch from your JSON file)
  useEffect(() => {
    setProducts(sampleData.results);
  }, []);

  // Calculate statistics
  const stats = useMemo(() => {
    const research = products.filter(p => p.analysis.verdict === 'RESEARCH FURTHER').length;
    const skip = products.filter(p => p.analysis.verdict === 'SKIP').length;
    const avgViability = products.reduce((sum, p) => sum + p.analysis.viability_score, 0) / products.length || 0;

    return { total: products.length, research, skip, avgViability };
  }, [products]);

  // Filter and sort products
  const displayProducts = useMemo(() => {
    let filtered = [...products];

    // Apply filters
    if (filterVerdict !== 'all') {
      filtered = filtered.filter(p => p.analysis.verdict === filterVerdict);
    }

    filtered = filtered.filter(p => p.analysis.viability_score >= minViability);

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'viability':
          return b.analysis.viability_score - a.analysis.viability_score;
        case 'profit':
          const profitA = a.analysis.estimated_amazon_price_range.high - a.price;
          const profitB = b.analysis.estimated_amazon_price_range.high - b.price;
          return profitB - profitA;
        case 'demand':
          return b.analysis.market_demand_score - a.analysis.market_demand_score;
        case 'quality':
          return b.analysis.product_quality_score - a.analysis.product_quality_score;
        default:
          return 0;
      }
    });

    return filtered;
  }, [products, filterVerdict, sortBy, minViability]);

  // Score color helper
  const getScoreColor = (score) => {
    if (score >= 8) return 'text-green-600 bg-green-100';
    if (score >= 6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  // Competition color helper
  const getCompetitionColor = (level) => {
    switch (level) {
      case 'Low': return 'text-green-600';
      case 'Medium': return 'text-yellow-600';
      case 'High': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const ProductCard = ({ product }) => {
    const profit = product.analysis.estimated_amazon_price_range.high - product.price;
    const roi = (profit / product.price * 100).toFixed(0);
    const isSelected = selectedProducts.includes(product.temu_id);

    return (
      <div className={`border rounded-lg p-4 hover:shadow-lg transition-shadow ${isSelected ? 'ring-2 ring-blue-500' : ''}`}>
        <div className="relative">
          <div className="h-48 bg-gray-200 rounded-md mb-3 flex items-center justify-center text-gray-500">
            Product Image
          </div>
          <div className="absolute top-2 left-2 bg-black text-white px-2 py-1 rounded text-sm">
            ${product.price}
          </div>
          <div className={`absolute top-2 right-2 px-2 py-1 rounded text-sm ${
            product.analysis.verdict === 'RESEARCH FURTHER' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
          }`}>
            {product.analysis.verdict === 'RESEARCH FURTHER' ? '✅' : '❌'}
          </div>
        </div>

        <h3 className="font-semibold text-sm mb-2 line-clamp-2">{product.title}</h3>

        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="text-center">
            <div className="text-xs text-gray-500">Viability</div>
            <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${getScoreColor(product.analysis.viability_score)}`}>
              {product.analysis.viability_score}/10
            </span>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Quality</div>
            <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${getScoreColor(product.analysis.product_quality_score)}`}>
              {product.analysis.product_quality_score}/10
            </span>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Demand</div>
            <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${getScoreColor(product.analysis.market_demand_score)}`}>
              {product.analysis.market_demand_score}/10
            </span>
          </div>
        </div>

        <div className="bg-green-50 p-2 rounded mb-3">
          <div className="text-xs font-semibold text-green-700">
            Est: ${product.analysis.estimated_amazon_price_range.low} - ${product.analysis.estimated_amazon_price_range.high}
          </div>
          <div className="text-xs text-gray-600">
            Profit: ${profit.toFixed(2)} (ROI: {roi}%)
          </div>
        </div>

        <div className="flex justify-between text-xs mb-3">
          <span className={getCompetitionColor(product.analysis.competition_level)}>
            Competition: {product.analysis.competition_level}
          </span>
          <span className={getCompetitionColor(product.analysis.safety_compliance_risks.risk_level)}>
            Risk: {product.analysis.safety_compliance_risks.risk_level}
          </span>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => {
              setSelectedProducts(prev =>
                prev.includes(product.temu_id)
                  ? prev.filter(id => id !== product.temu_id)
                  : [...prev, product.temu_id]
              );
            }}
            className="flex-1 bg-blue-500 text-white py-1 px-2 rounded text-xs hover:bg-blue-600"
          >
            {isSelected ? 'Remove' : 'Compare'}
          </button>
          <button className="flex-1 bg-gray-200 py-1 px-2 rounded text-xs hover:bg-gray-300">
            Details
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">Temu Product Analysis Dashboard</h1>

        <div className="grid grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{stats.total}</div>
              <div className="text-sm text-gray-500">Total Products</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-green-600">{stats.research}</div>
              <div className="text-sm text-gray-500">Research Further</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-red-600">{stats.skip}</div>
              <div className="text-sm text-gray-500">Skip</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{stats.avgViability.toFixed(1)}</div>
              <div className="text-sm text-gray-500">Avg Viability</div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="block text-sm font-medium mb-1">Sort By</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="border rounded px-3 py-1"
            >
              <option value="viability">Viability Score</option>
              <option value="profit">Profit</option>
              <option value="demand">Market Demand</option>
              <option value="quality">Product Quality</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Filter Verdict</label>
            <select
              value={filterVerdict}
              onChange={(e) => setFilterVerdict(e.target.value)}
              className="border rounded px-3 py-1"
            >
              <option value="all">All</option>
              <option value="RESEARCH FURTHER">Research Further</option>
              <option value="SKIP">Skip</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Min Viability Score</label>
            <input
              type="number"
              min="0"
              max="10"
              value={minViability}
              onChange={(e) => setMinViability(Number(e.target.value))}
              className="border rounded px-3 py-1 w-20"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">View Mode</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value)}
              className="border rounded px-3 py-1"
            >
              <option value="grid">Grid</option>
              <option value="list">List</option>
            </select>
          </div>
        </div>
      </div>

      {/* Product List */}
      {displayProducts.length === 0 ? (
        <Alert variant="destructive">
          <AlertDescription>No products match your filters.</AlertDescription>
        </Alert>
      ) : (
        <div className={viewMode === 'grid' ? "grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6" : "space-y-4"}>
          {displayProducts.map(product => (
            <ProductCard key={product.temu_id} product={product} />
          ))}
        </div>
      )}

      {/* Comparison Section */}
      {selectedProducts.length > 0 && (
        <div className="mt-8 p-4 bg-gray-50 rounded shadow">
          <h2 className="text-xl font-semibold mb-4">Comparison</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-300">
              <thead>
                <tr>
                  <th className="border border-gray-300 p-2">Title</th>
                  <th className="border border-gray-300 p-2">Price</th>
                  <th className="border border-gray-300 p-2">Viability</th>
                  <th className="border border-gray-300 p-2">Quality</th>
                  <th className="border border-gray-300 p-2">Demand</th>
                  <th className="border border-gray-300 p-2">Verdict</th>
                  <th className="border border-gray-300 p-2">Competition</th>
                  <th className="border border-gray-300 p-2">Risk</th>
                  <th className="border border-gray-300 p-2">Profit</th>
                  <th className="border border-gray-300 p-2">ROI %</th>
                </tr>
              </thead>
              <tbody>
                {products.filter(p => selectedProducts.includes(p.temu_id)).map(product => {
                  const profit = product.analysis.estimated_amazon_price_range.high - product.price;
                  const roi = (profit / product.price * 100).toFixed(0);
                  return (
                    <tr key={product.temu_id} className="text-center">
                      <td className="border border-gray-300 p-2 text-left">{product.title}</td>
                      <td className="border border-gray-300 p-2">${product.price.toFixed(2)}</td>
                      <td className={`border border-gray-300 p-2 ${getScoreColor(product.analysis.viability_score)}`}>{product.analysis.viability_score}</td>
                      <td className={`border border-gray-300 p-2 ${getScoreColor(product.analysis.product_quality_score)}`}>{product.analysis.product_quality_score}</td>
                      <td className={`border border-gray-300 p-2 ${getScoreColor(product.analysis.market_demand_score)}`}>{product.analysis.market_demand_score}</td>
                      <td className="border border-gray-300 p-2">{product.analysis.verdict}</td>
                      <td className={`border border-gray-300 p-2 ${getCompetitionColor(product.analysis.competition_level)}`}>{product.analysis.competition_level}</td>
                      <td className={`border border-gray-300 p-2 ${getCompetitionColor(product.analysis.safety_compliance_risks.risk_level)}`}>{product.analysis.safety_compliance_risks.risk_level}</td>
                      <td className="border border-gray-300 p-2">${profit.toFixed(2)}</td>
                      <td className="border border-gray-300 p-2">{roi}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <button
            onClick={() => setSelectedProducts([])}
            className="mt-4 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Clear Comparison
          </button>
        </div>
      )}
    </div>
  );
};

export default ProductDashboard;
