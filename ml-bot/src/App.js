
import './App.css';
import "bootstrap/dist/css/bootstrap.min.css";
import Form from "./components/form.js";
import Navbar from './components/NavBar';
import Home from "./components/Home.component";
import {BrowserRouter as Router,Routes,Route} from "react-router-dom";
import Result from './components/Results.js';
function App() {
  return (
    <Router>
      <Navbar/>
      <Routes>
        <Route exact path="/" element={<Form/>}/>
        <Route exact path="/result" element={<Result/>}/>
      </Routes>
      
    </Router>
    
  );
}

export default App;
