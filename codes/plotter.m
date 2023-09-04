function plotter(filename)
load(filename)
alphabet = 'abcdefgh';
methods = length(errors_original(:,1,1));
measures = length(errors_original(1,:,1));
fig = 0;

for method = 1:methods
    
    for measure = 1:measures
        
        fig = fig + 1;
        figure(fig)
        semilogy((1:length(errors_original(method,measure,:)))-1,reshape(errors_projection(method,measure,:),[length(errors_function(method,measure,:)) 1 1]),'k-*','LineWidth',4)
        hold on
        semilogy((1:length(errors_original(method,measure,:)))-1,reshape(errors_original(method,measure,:),[length(errors_original(method,measure,:)) 1 1]),'b-*','LineWidth',3)
        semilogy((1:length(errors_original(method,measure,:)))-1,reshape(errors_function(method,measure,:),[length(errors_function(method,measure,:)) 1 1]),'r-*','LineWidth',2)
        xlabel('$q$','Interpreter','latex')
        ylabel('$\varepsilon$','Interpreter','latex')
        legend({'Projection','Nyström','funNyström'},'Location','best')
        set(gca,'Fontsize',18)
        hold off
        figname = append('fig',num2str(method),alphabet(measure));
        print(figname,'-depsc')
        
    end
    
end
end