import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    amount: 100,
    location: 'NY',
    device_type: 'Mobile',
    merchant_category: 'Electronics',
    transaction_hour: 12,
    past_transaction_count: 5,
    avg_spending: 50
  });

  const [prediction, setPrediction] = useState(null);
  const [reportImage, setReportImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      // Convert types
      const payload = {
        ...formData,
        amount: parseFloat(formData.amount),
        transaction_hour: parseInt(formData.transaction_hour),
        past_transaction_count: parseInt(formData.past_transaction_count),
        avg_spending: parseFloat(formData.avg_spending)
      };

      const response = await axios.post('http://localhost:5000/predict', payload);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to connect to backend.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateGraph = async () => {
    setLoading(true);
    setError(null);
    try {
      // Convert types
      const payload = {
        ...formData,
        amount: parseFloat(formData.amount),
        transaction_hour: parseInt(formData.transaction_hour),
        past_transaction_count: parseInt(formData.past_transaction_count),
        avg_spending: parseFloat(formData.avg_spending)
      };

      const response = await axios.post('http://localhost:5000/visualize', payload);
      setReportImage(response.data.image);
    } catch (err) {
      setError('Failed to generate graph.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glass-panel">
        <h1 className="title">AI FRAUD DETECTOR</h1>

        <div className="content-grid">
          {/* Input Section */}
          <div className="input-section">
            <h2><span style={{ fontSize: '1.2em' }}>⚡</span> Transaction Data</h2>

            <div className="form-group">
              <label>Amount (USD)</label>
              <input type="number" name="amount" value={formData.amount} onChange={handleChange} placeholder="e.g. 500.00" />
            </div>

            <div className="form-group">
              <label>Location</label>
              <select name="location" value={formData.location} onChange={handleChange}>
                <option value="NY">New York (NY)</option>
                <option value="CA">California (CA)</option>
                <option value="TX">Texas (TX)</option>
                <option value="FL">Florida (FL)</option>
                <option value="IL">Illinois (IL)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Device Type</label>
              <select name="device_type" value={formData.device_type} onChange={handleChange}>
                <option value="Mobile">Mobile Device</option>
                <option value="Desktop">Desktop / Laptop</option>
                <option value="Tablet">Tablet</option>
              </select>
            </div>

            <div className="form-group">
              <label>Merchant Category</label>
              <select name="merchant_category" value={formData.merchant_category} onChange={handleChange}>
                <option value="Electronics">Electronics</option>
                <option value="Groceries">Groceries</option>
                <option value="Clothing">Clothing</option>
                <option value="Restaurant">Restaurant</option>
                <option value="Travel">Travel</option>
              </select>
            </div>

            <div className="form-group">
              <label>Hour of Day (24h)</label>
              <input type="number" name="transaction_hour" value={formData.transaction_hour} onChange={handleChange} min="0" max="23" />
            </div>

            <div className="form-group">
              <label>Past Transaction Count</label>
              <input type="number" name="past_transaction_count" value={formData.past_transaction_count} onChange={handleChange} />
            </div>

            <div className="form-group">
              <label>Avg Spending ($)</label>
              <input type="number" name="avg_spending" value={formData.avg_spending} onChange={handleChange} />
            </div>

            <div className="button-group">
              <button className="btn-primary" onClick={handlePredict} disabled={loading}>
                {loading ? 'Analyzing...' : 'DETECT FRAUD'}
              </button>
              <button className="btn-secondary" onClick={handleGenerateGraph} disabled={loading}>
                {loading ? 'Creating...' : 'VIEW REPORT'}
              </button>
            </div>
          </div>

          {/* Results Section */}
          <div className="results-section">
            <h2><span style={{ fontSize: '1.2em' }}>📊</span> Live Analysis</h2>
            {error && <div className="error-msg">{error}</div>}

            {prediction && (
              <div className={`prediction-card ${prediction.is_high_risk ? 'risk-high' : 'risk-low'}`}>
                <div className="status">{prediction.is_high_risk ? 'CRITICAL RISK DETECTED' : 'TRANSACTION SAFE'}</div>
                <div className="score">{prediction.probability_pct.toFixed(2)}%</div>
                <div style={{ color: 'var(--text-muted)' }}>Probability of Fraud</div>
              </div>
            )}

            {reportImage && (
              <div className="report-image-container">
                <img src={`data:image/png;base64,${reportImage}`} alt="Fraud Report" />
                <a href={`data:image/png;base64,${reportImage}`} download="report.png" className="download-link"> Download High-Res Report </a>
              </div>
            )}

            {!prediction && !reportImage && !loading && (
              <div className="placeholder-text">
                <div>
                  Enter transaction parameters on the left to begin real-time analysis.
                </div>
              </div>
            )}

            {loading && (
              <div className="placeholder-text">
                <div style={{ color: 'var(--primary)' }}>Processing neural network request...</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
