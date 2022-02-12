import React from 'react';
import axios from "axios";
class NewComponent extends React.Component {
    constructor(props){
        super(props);
        this.state={
                data:[]
        };
    }
    componentDidMount(){
         axios.get('http://localhost:8000/getData')
        .then((response) => {
          console.log(response.data);
          this.setState({
            data: response.data
          });
        });
    }
    render(){
        return(
       <>
       <h1>Results</h1>
 {
    this.state.data.map((item,index)=>{
       return(
        <div className="card">
        <div className="card-header">
          Featured
        </div>
        <div className="card-body">
          <h5 className="card-title">Hiiii{item.modelName}</h5>
          <p className="card-text"></p>
          <h2>{item.accuracy}</h2>
        </div>
        </div> 
             
       );
       }
   )
}
       </>      
        );
}
}
  export default NewComponent;


 

