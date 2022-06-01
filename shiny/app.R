library(shiny)
library(dplyr)
library(ggplot2)
library(kableExtra)
library(nnfR)

# Define UI for app that draws a histogram ----
ui = fluidPage(
  
  # App title ----
  titlePanel("Multi-Class Classification"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of data points per class ----
      sliderInput(inputId = "data_points",
                  label = "Data points per class",
                  min = 1,
                  max = 100,
                  value = 50),
      # Input: Slide for the number of classes in data
      sliderInput(inputId = "num_class",
                  label = "Number of Classes",
                  min = 2,
                  max = 10,
                  value = 3)
    ),
    # Main panel for displaying plot outputs ----
    mainPanel(
      # Output: Spiral Data plot ----
      plotOutput(outputId = "spiral_plot")
    )
  ),
  
  # Sidebar layout with inputs for model ----
  sidebarLayout(
    
    # Side bar panel for model inputs & params ----
    sidebarPanel(
    h4("Layer 1"),
    sliderInput(inputId = "num_neurons",
                label = "Number of Neurons",
                min = 1,
                max = 5,
                value = 3),
    sliderInput(inputId = "dropout",
                label = "Dropout rate",
                min = 0,
                max = 0.9,
                value = 0)
    ),
    # Main panel for displaying model
    mainPanel(
      shiny::uiOutput(outputId = "model"),
    )
  ),
  
  # Sidebar layout with training inputs ---- 
  sidebarLayout(
    #Side bar panel for epochs ----
    sidebarPanel(
      h4("Train"),
      sliderInput(inputId = "epochs",
                  label = "Number of Training Epochs",
                  min = 1,
                  max = 100,
                  value = 10)
    ),
    # Main panel for displaying the training results
    mainPanel(
      shiny::plotOutput(outputId = "train_results"),
      shiny::uiOutput(outputId = "conf_mat"),
      shiny::uiOutput(outputId = "performance")
    )
  )
)

server = function(input, output) {

  # Plotting the Task ----
  # 1. It is "reactive" and therefore should be automatically
  #    re-executed when inputs (input$bins) change
  # 2. Its output type is a plot
  sim_spiral = reactive({
    spiral_data = nnfR::sim_spiral_data(N = input$data_points, 
                                        K = input$num_class)
    if (input$num_class <= 2) spiral_data$label = spiral_data$label - 1
    return(spiral_data)
  })
  
  build_model = reactive({
    spiral_data = sim_spiral()
    model = nnfR::initialize_model() %>%
      #layer1
      nnfR::add_dense(n_inputs = ncol(spiral_data[, c("x", "y")]),
                      n_neurons = input$num_neurons,
                      dropout_rate = input$dropout
                      ) %>%
      nnfR::add_activation("relu") %>%
      #output layer
      nnfR::add_dense(n_inputs = input$num_neurons,
                      n_neurons = input$num_class) %>%
      nnfR::add_activation("softmax_crossentropy") %>%
      #loss function
      nnfR::add_loss("categorical_crossentropy")
  })
  
  #Outputs:
  #spiral data plot
  output$spiral_plot = renderPlot({
    
    spiral_data = sim_spiral()
    plot(spiral_data$x, spiral_data$y, col = spiral_data$label, 
         border = "white",
         xlab = "x", ylab = "y",
         main = "Simulated Spiral Data")
  })
  
  # Building Model ----
  output$model = renderTable({
    
    # spiral_data = nnfR::sim_spiral_data(N = input$data_points, 
    #                                    K = input$num_class)
    spiral_data = sim_spiral()
    X = as.matrix(spiral_data[,c("x","y")])
    y = spiral_data$label
    
    model = build_model()
    model_info = tibble(
      "Layer" = c("Layer 1", "Output Layer"),
      "Number of Inputs" = c("2 (x & y coordinates)", input$num_neurons),
      "Number of Neurons" = c(input$num_neurons, input$num_class),
      "Dropout Rate" = c(input$dropout, 0),
      "Activation Function" = c("reLU", "softmax"),
      "Loss Function" = c(" -- ", "categorical crossentropy")
    )
    
    print(model_info)
  })
  
  #Train the model given the customized inputs
  fit_model = reactive({

    ##Data
    spiral_data = sim_spiral()
    X = as.matrix(spiral_data[,c("x","y")])
    y = spiral_data$label
    ##Build Model
    model = build_model()
    ##Train Model
    fit = nnfR::train_model(model = model,
                            inputs = X,
                            y_true = y,
                            epochs = input$epochs,
                            optimizer = "adam",
                            learning_rate = 0.01,
                            lr_decay = 1e-3,
                            beta_1 = 0.5, beta_2 = 0.5,
                            metric_list = c("accuracy"),
                            epoch_print = input$epochs)
    return(fit)
  })
  
  #Plot loss
  output$train_results = renderPlot({
    fit = fit_model()
    
    plot(fit$metrics$epoch, fit$metrics$loss,
         type = "l", col = "red", xlab = "epoch", ylab = "loss")
    # lines(fit$metrics$epoch, fit$metrics$validation_loss,
    #       type = "l", col = "blue")
    legend(x = "topright", legend = c("training", "validation"),
           lty = c(1, 1), col = c("red", "blue"))
  })
  
  #Calculate performance metrics
  metrics = reactive({
    spiral_data  = sim_spiral()
    fit = fit_model()
    
    perf = classess(truths = spiral_data$label,
                    predictions = fit$predictions)
    return(perf)
  })
  
  #View confusion matrix
  output$conf_mat = renderTable({
    perf = metrics()
    data = sim_spiral()
    
    conf_mat = perf$conf_mat
    # conf_mat = matrix(conf_mat, nrow(conf_mat), ncol(conf_mat)
    #                   # dimnames = list(Pred=c(unique(data$label)),
    #                   #                 Truth=c(unique(data$label))
    #                   # )
    # )
    # labels = unique(data$label[1])
    # colnames(conf_mat) = c(paste("True:",labels[1]),
    #                              labels[-1])
    # rownames(conf_mat) = c(paste("Pred:",labels[1]),
    #                        labels[-1])
    print(conf_mat)
    
  })
  
  #View performance metrics
  output$performance = renderTable({
    perf = metrics()
    print(perf$metrics)
  })
  
}




shinyApp(ui = ui, server = server)




