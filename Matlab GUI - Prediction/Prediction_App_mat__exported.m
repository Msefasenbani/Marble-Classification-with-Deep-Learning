classdef Prediction_App_mat__exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure            matlab.ui.Figure
        UIAxes              matlab.ui.control.UIAxes
        LOADIMAGEButton     matlab.ui.control.Button
        PredictionLabel     matlab.ui.control.Label
        AccuracyLabel       matlab.ui.control.Label
        PredictionAppLabel  matlab.ui.control.Label
    end

    
    properties (Access = public)
        net
        sz
        % Description
    end
    
    properties (Access = private)
        Property2 % Description
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
             app.net = load('Xception_Model_1.mat');
             app.sz = app.net.layers(1).InputSize ;
        end

        % Button pushed function: LOADIMAGEButton
        function LOADIMAGEButtonPushed(app, event)
             
            
             
             [path,file]=uigetfile({'*.jpg; *.bmp;*.gif;*.tiff;*.Tiff;*.Tif;*.tiff;*.jfif;*.jpeg;*.png'}, 'Select file');
             Picture=[file path];
             OrginalPic=imread(Picture);
             
             
             %OrginalPic = OrginalPic(1:app.sz(1),1:app.sz(2),1:app.sz(3));
             OrginalPic = imresize(  OrginalPic ,  [299,299]) ;
             imshow(OrginalPic,'Parent',app.UIAxes);
            
             
             
             [label,score] = classify(app.net.net, OrginalPic);  
             app.PredictionLabel.Text = char(label);
             app.AccuracyLabel.Text = num2str(max(score)*100);   
             
             
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.9294 0.6941 0.1255];
            app.UIFigure.Colormap = [0.2431 0.149 0.6588;0.251 0.1647 0.7059;0.2588 0.1804 0.7529;0.2627 0.1961 0.7961;0.2706 0.2157 0.8353;0.2745 0.2353 0.8706;0.2784 0.2549 0.898;0.2784 0.2784 0.9216;0.2824 0.302 0.9412;0.2824 0.3216 0.9569;0.2784 0.3451 0.9725;0.2745 0.3686 0.9843;0.2706 0.3882 0.9922;0.2588 0.4118 0.9961;0.2431 0.4353 1;0.2196 0.4588 0.9961;0.1961 0.4863 0.9882;0.1843 0.5059 0.9804;0.1804 0.5294 0.9686;0.1765 0.549 0.9529;0.1686 0.5686 0.9373;0.1529 0.5922 0.9216;0.1451 0.6078 0.9098;0.1373 0.6275 0.898;0.1255 0.6471 0.8902;0.1098 0.6627 0.8745;0.0941 0.6784 0.8588;0.0706 0.6941 0.8392;0.0314 0.7098 0.8157;0.0039 0.7216 0.7922;0.0078 0.7294 0.7647;0.0431 0.7412 0.7412;0.098 0.749 0.7137;0.1412 0.7569 0.6824;0.1725 0.7686 0.6549;0.1922 0.7765 0.6235;0.2157 0.7843 0.5922;0.2471 0.7922 0.5569;0.2902 0.7961 0.5176;0.3412 0.8 0.4784;0.3922 0.8039 0.4353;0.4471 0.8039 0.3922;0.5059 0.8 0.349;0.5608 0.7961 0.3059;0.6157 0.7882 0.2627;0.6706 0.7804 0.2235;0.7255 0.7686 0.1922;0.7725 0.7608 0.1647;0.8196 0.749 0.1529;0.8627 0.7412 0.1608;0.902 0.7333 0.1765;0.9412 0.7294 0.2118;0.9725 0.7294 0.2392;0.9961 0.7451 0.2353;0.9961 0.7647 0.2196;0.9961 0.7882 0.2039;0.9882 0.8118 0.1882;0.9804 0.8392 0.1765;0.9686 0.8627 0.1647;0.9608 0.8902 0.1529;0.9608 0.9137 0.1412;0.9647 0.9373 0.1255;0.9686 0.9608 0.1059;0.9765 0.9843 0.0824];
            app.UIFigure.Position = [100 100 1079 708];
            app.UIFigure.Name = 'UI Figure';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            app.UIAxes.LineWidth = 3;
            app.UIAxes.Visible = 'off';
            app.UIAxes.BackgroundColor = [0.9294 0.6941 0.1255];
            app.UIAxes.Position = [294 118 495 442];

            % Create LOADIMAGEButton
            app.LOADIMAGEButton = uibutton(app.UIFigure, 'push');
            app.LOADIMAGEButton.ButtonPushedFcn = createCallbackFcn(app, @LOADIMAGEButtonPushed, true);
            app.LOADIMAGEButton.BackgroundColor = [1 1 0];
            app.LOADIMAGEButton.FontSize = 20;
            app.LOADIMAGEButton.FontWeight = 'bold';
            app.LOADIMAGEButton.FontColor = [0 0.4471 0.7412];
            app.LOADIMAGEButton.Position = [42 314 211 83];
            app.LOADIMAGEButton.Text = 'LOAD IMAGE';

            % Create PredictionLabel
            app.PredictionLabel = uilabel(app.UIFigure);
            app.PredictionLabel.BackgroundColor = [1 1 0];
            app.PredictionLabel.HorizontalAlignment = 'center';
            app.PredictionLabel.FontSize = 26;
            app.PredictionLabel.FontWeight = 'bold';
            app.PredictionLabel.FontColor = [0 0.451 0.7412];
            app.PredictionLabel.Position = [798 352 277 84];
            app.PredictionLabel.Text = 'Prediction';

            % Create AccuracyLabel
            app.AccuracyLabel = uilabel(app.UIFigure);
            app.AccuracyLabel.BackgroundColor = [1 1 0];
            app.AccuracyLabel.HorizontalAlignment = 'center';
            app.AccuracyLabel.FontSize = 26;
            app.AccuracyLabel.FontWeight = 'bold';
            app.AccuracyLabel.FontColor = [0 0.451 0.7412];
            app.AccuracyLabel.Position = [826 238 219 74];
            app.AccuracyLabel.Text = 'Accuracy';

            % Create PredictionAppLabel
            app.PredictionAppLabel = uilabel(app.UIFigure);
            app.PredictionAppLabel.FontSize = 30;
            app.PredictionAppLabel.FontWeight = 'bold';
            app.PredictionAppLabel.FontColor = [0 0.4471 0.7412];
            app.PredictionAppLabel.Position = [426 648 230 53];
            app.PredictionAppLabel.Text = 'Prediction App';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Prediction_App_mat__exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end